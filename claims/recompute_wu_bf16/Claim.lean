import TVA

/-!
# `recompute_wu_bf16`: the prepared claim

Source: `source/wy_fast.py`. The kernel is `recompute_w_u_fwd_kernel` (lines 69–133; modelled once per dtype of
`beta`, `recompute_w_u_fwd_kernel_b32` and `_b16`) with the helper
`exp2` (lines 65–66) it inlines; it reads the module-level names `triton` and `tl` (lines 61–62).
The host `recompute_w_u_fwd` (lines 175–220) launches it on `bfloat16` `k` (`[B, T, H, K]`), `v`
(`[B, T, HV, V]`) and `A` (`[B, T, HV, 64]`), `float32` or `bfloat16` `beta` and `float32` `g` (`[B, T, HV]`), into fresh
`bfloat16` `w` (`[B, T, HV, K]`) and `u` (`[B, T, HV, V]`), over a padded batch (`cu_seqlens = None`)
or packed variable-length sequences (`cu_seqlens` and `chunk_indices`, `int32` or `int64`), with
`g` given (`USE_G = True`), as the gated DeltaNet's chunked forward and backward and the gated
delta-rule layers built on them call it.

The claim (`Statement`): on every covered instance and input, every launch ends normally; for every
token `t` of every sequence, every value head `h`, with the token's chunk the 64-token block of its
sequence that holds it and `A_s` the entries of its row of `A`:

* every `u[t, h, c]` is a finite `bfloat16` within `tolU` of `Σ_s A_s · vb_s`, `vb_s` the kernel's own
  `bfloat16` operand of line 115 (the `bfloat16` rounding of the `float32` product
  `v[s, h, c] · beta[s, h]` for a `float32` `beta`, the `bfloat16` product for a `bfloat16` one: IEEE
  operations, exact functions of the inputs), the dot product's error and one rounding to
  `bfloat16`;
* every `w[t, h, c]` is a finite `bfloat16` within `tolW` of the exact
  `Σ_s A_s · k[s, h / (HV / H), c] · beta[s, h] · 2^(g[s, h])`;

`s` running over the chunk's tokens inside the sequence; and every byte outside `w` and `u` is
unchanged.

This file holds the model (`recompute_w_u_fwd_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`Post`, `Spec`), the assumptions (`Assumptions`), the claim
(`Statement`), and the proofs that it is not vacuous (`consistent`, `satisfiable`) and that it covers
the setup's launches (`setup_*_ok`). It does not prove `Statement`.
-/

namespace Claims.RecomputeWU

open TVA TVA.Math.Float

/-- The configuration of line 116's `tl.dot(b_A, b_vb, allow_tf32=False)`: `bfloat16` operands, the
default `float32` result, `allow_tf32=False` (`input_precision = "ieee"`), the default
`max_num_imprecise_acc` (`0` for a 16-bit dot). -/
def cfgU : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .ieee, maxImpreciseAcc := 0 }

/-- The configuration of line 132's `tl.dot(b_A, b_kb.to(b_k.dtype))`: as `cfgU`, with the default
`input_precision` (`tf32`; Triton ignores it for `bfloat16` operands). -/
def cfgW : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The model -/

/-- `recompute_w_u_fwd_kernel` (source lines 69–133) for a `float32` `beta`, with `USE_G = True`, the
value the host passes on every covered launch (`g` is given), so the two `if USE_G:` blocks (lines 119–121, 130–131) are
part of the body and the model has no parameter for the flag. `k`, `v`, `w`, `u`, `A` are `*bf16`
pointers, `beta`, `g` `*fp32`; `cu_seqlens` and `chunk_indices` are pointers of one integer dtype `sD`
(`int32` or `int64`; the host builds `chunk_indices` with `.to(cu_seqlens)`). They are `None` on the
padded path, where the model's `IS_VARLEN = false` branch never uses them. `T`
(`do_not_specialize=['T']`: a runtime `int32` even at `1`) is `I32`. `H HV K V BT BK BV` are the
`tl.constexpr` parameters, Lean parameters; `IS_VARLEN` is a `Bool` parameter. -/
def recompute_w_u_fwd_kernel_b32 {sD : DType} (k v : Ptr .bf16) (beta : Ptr .f32) (w u A : Ptr .bf16)
    (g : Ptr .f32) (cu_seqlens chunk_indices : Ptr sD) (T : I32)
    (H HV K V BT BK BV : Nat) (IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 91: axes 0 and 1 converted to `int64` (`.to(tl.int64)`, sign extension)
    let i_t0 : I64 := IntV.cast (tl.program_id ctx 0)
    let i_bh : I64 := IntV.cast (tl.program_id ctx 1)
    -- line 92: `i_bh // HV`, `i_bh % HV`, left to right: the constexpr `HV` built in `int64` (the
    -- tensor's dtype); truncating `//` and `%` (`tl.idiv`, `tl.irem`)
    let i_b : I64 := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    let i_h : I64 := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    -- lines 93–98: the compile-time `if IS_VARLEN`; the front end visits the taken branch only.
    -- Each branch gives the program's chunk `i_t`, the sequence's first token `bos` and the
    -- sequence length `T` (line 96 rebinds `T` to the `int64` `eos - bos`; the padded `T` is the
    -- `int32` argument, sign-extended where an `int64` operand meets it)
    let st ← (if IS_VARLEN then (do
        -- line 94: `i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)` (`i_t * 2`: the constant
        -- `2` in `int64`), then `i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)` with the
        -- old `i_t` (the right side is evaluated first; `+ 1`: the pointer plus the `int32` `1`)
        let v_n ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64))
        let i_n : I32 := (← tl.convert sD .i32 (Tensor.scalar v_n)).item
        let v_t ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
        let i_t : I64 := (← tl.convert sD .i64 (Tensor.scalar v_t)).item
        -- line 95: `bos = tl.load(cu_seqlens + i_n).to(tl.int64)`,
        -- `eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)`
        let v_b ← tl.load1 (cu_seqlens + i_n)
        let bos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_b)).item
        let v_e ← tl.load1 (cu_seqlens + i_n + (IntV.ofNat 1 : I32))
        let eos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_e)).item
        -- line 96: `T = eos - bos` (`int64`)
        let T_n : I64 := eos - bos
        pure (i_t, bos, T_n))
      else (do
        -- line 98: `bos, eos = i_b * T, i_b * T + T`: `int64` products and sum (`T` sign-extended);
        -- `eos` is not read on this path
        let bos : I64 := i_b * (IntV.cast T : I64)
        let _eos : I64 := i_b * (IntV.cast T : I64) + (IntV.cast T : I64)
        pure (i_t0, bos, (IntV.cast T : I64))) : Prog (I64 × I64 × I64))
    let (i_t, bos, T_n) := st
    -- line 99: `i_t * BT + tl.arange(0, BT)`: the constexpr in `int64`, the `int32` range
    -- sign-extended
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + ((tl.arange 0 BT).castInt : Tensor [BT] I64)
    -- line 100
    let o_A : Tensor [BT] I32 := tl.arange 0 BT
    -- line 101: the signed `int64` comparison (`T` is `int64`, or the `int32` argument
    -- sign-extended)
    let m_t : Tensor [BT] Bool := o_t.ltS T_n
    -- line 102: `m_t[:, None] & (o_A[None, :] < BT)`: the constexpr an `int32` tensor of its own
    -- type (a comparison), the broadcast to `[BT, BT]` written at once
    let m_A : Tensor [BT, BT] Bool := (m_t.col BT).and ((o_A.ltS (IntV.ofNat BT : I32)).row BT)
    -- line 103: `((beta + bos*HV) + i_h) + o_t * HV`: `int64` scalars added to the pointer, then
    -- the `int64` block (the constexpr `HV` in `int64`)
    let p_b : Tensor [BT] (Ptr .f32) :=
      beta + bos * (IntV.ofNat HV : I64) + i_h + o_t * (IntV.ofNat HV : I64)
    -- line 104: `other=0.0` is the `float32` `+0`, the pointer's dtype
    let b_b : Tensor [BT] F32 ← tl.load p_b (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- line 106: `((A + (bos*HV + i_h) * BT) + o_t[:, None] * (HV*BT)) + o_A[None, :]`: the `int64`
    -- scalar, then the `int64` column (`HV*BT` computed by Python, built in `int64`), then the
    -- `int32` row
    let p_A : Tensor [BT, BT] (Ptr .bf16) :=
      A + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat BT : I64) +
        o_t.col BT * (IntV.ofNat (HV * BT) : I64) + o_A.row BT
    -- line 107: `other=0.0`: the `float32` `+0` converted to the pointer's `bfloat16`
    let zA : Tensor [BT, BT] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BT] (FloatV.litF32 (FloatV.ofPat 0)))
    let b_A ← tl.load p_A (some m_A) (some zA)
    -- line 109: `range(tl.cdiv(V, BV))`: the bound is Python's `(V + (BV - 1)) // BV` (two
    -- constexprs, evaluated twice with no operation), an `int32` constant, so `i_v` is an `int32`
    -- from `0`, step `1`; the loop carries no name (every name the body assigns is first defined
    -- there and not used after the loop)
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((V + (BV - 1)) / BV) : I32) (IntV.ofInt 1 : I32) ()
        fun i_v _ => do
      -- line 110: the constexpr `BV` built in `int32`
      let o_v : Tensor [BV] I32 := i_v * (IntV.ofNat BV : I32) + tl.arange 0 BV
      -- line 111: `m_t[:, None] & (o_v[None, :] < V)`: `V` an `int32` tensor (`Inst.ok.V_lt`)
      let m_v : Tensor [BT, BV] Bool := (m_t.col BV).and ((o_v.ltS (IntV.ofNat V : I32)).row BT)
      -- line 112: as line 106, with `V` and `HV*V`
      let p_v : Tensor [BT, BV] (Ptr .bf16) :=
        v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
          o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
      -- line 113: as line 112
      let p_u : Tensor [BT, BV] (Ptr .bf16) :=
        u + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
          o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
      -- line 114: `other=0.0` as line 107
      let zV : Tensor [BT, BV] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BV] (FloatV.litF32 (FloatV.ofPat 0)))
      let b_v ← tl.load p_v (some m_v) (some zV)
      -- line 115: `b_v * b_b[:, None]`: `bfloat16` times `float32`, so `b_v` is converted to `float32` (exact) and
      -- the product is the IEEE `float32` product (no addition consumes it); `.to(b_v.dtype)` one
      -- rounding to `bfloat16`
      let b_vf : Tensor [BT, BV] F32 ← tl.cvt b_v
      let b_vb : Tensor [BT, BV] BF16 ← tl.cvt (← tl.mul b_vf (b_b.col BV))
      -- line 116: `tl.dot` with the default accumulator (`float32` `+0`) and no addition after it
      let b_u ← tl.dot cfgU b_A b_vb (tl.zeros [BT, BV] .f32)
      -- line 117: `.to(p_u.dtype.element_ty)`: one rounding to `bfloat16`
      let s_u : Tensor [BT, BV] BF16 ← tl.cvt b_u
      tl.store p_u s_u (some m_v)
    -- line 119: `if USE_G:` is a compile-time `if`, taken (`USE_G = True`)
    -- line 120: `(g + (bos*HV + i_h)) + o_t * HV`: the parenthesised `int64` sum first
    let p_g : Tensor [BT] (Ptr .f32) :=
      g + (bos * (IntV.ofNat HV : I64) + i_h) + o_t * (IntV.ofNat HV : I64)
    -- line 121: `other=0.0` the `float32` `+0`; the helper `exp2` (lines 65–66) inlined: its
    -- `x.to(tl.float32)` is the identity on `float32`, then `tl.math.exp2`
    let l_g : Tensor [BT] F32 ← tl.load p_g (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    let b_g : Tensor [BT] F32 ← tl.exp2 l_g
    -- line 123: as line 109, with `K` and `BK`
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + (BK - 1)) / BK) : I32) (IntV.ofInt 1 : I32) ()
        fun i_k _ => do
      -- line 124
      let o_k : Tensor [BK] I32 := i_k * (IntV.ofNat BK : I32) + tl.arange 0 BK
      -- line 125: as line 111, with `K` (`Inst.ok.K_lt`)
      let m_k : Tensor [BT, BK] Bool := (m_t.col BK).and ((o_k.ltS (IntV.ofNat K : I32)).row BT)
      -- line 126: `(bos*H + i_h // (HV // H)) * K`: `HV // H` computed by Python, built in `int64`,
      -- the truncating `//` (`tl.idiv`); then as line 106 with `K` and `H*K`
      let p_k : Tensor [BT, BK] (Ptr .bf16) :=
        k + (bos * (IntV.ofNat H : I64) +
            (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
            (IntV.ofNat K : I64) +
          o_t.col BK * (IntV.ofNat (H * K) : I64) + o_k.row BT
      -- line 127: as line 106, with `K` and `HV*K`
      let p_w : Tensor [BT, BK] (Ptr .bf16) :=
        w + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) +
          o_t.col BK * (IntV.ofNat (HV * K) : I64) + o_k.row BT
      -- line 128: `other=0.0` as line 107
      let zK : Tensor [BT, BK] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BK] (FloatV.litF32 (FloatV.ofPat 0)))
      let b_k ← tl.load p_k (some m_k) (some zK)
      -- line 129: `b_k * b_b[:, None]`: `b_k` converted to `float32` (exact), the IEEE `float32` product
      let b_kf : Tensor [BT, BK] F32 ← tl.cvt b_k
      let b_kb ← tl.mul b_kf (b_b.col BK)
      -- line 130: `if USE_G:` taken; line 131: `b_kb *= b_g[:, None]` is `b_kb = b_kb * b_g[:, None]`,
      -- the IEEE `float32` product
      let b_kb ← tl.mul b_kb (b_g.col BK)
      -- line 132: `b_kb.to(b_k.dtype)`: one rounding to `bfloat16`; `tl.dot` with the default
      -- accumulator and no addition after it
      let b_kbh : Tensor [BT, BK] BF16 ← tl.cvt b_kb
      let b_w ← tl.dot cfgW b_A b_kbh (tl.zeros [BT, BK] .f32)
      -- line 133: one rounding to `bfloat16`
      let s_w : Tensor [BT, BK] BF16 ← tl.cvt b_w
      tl.store p_w s_w (some m_k)


/-- `recompute_w_u_fwd_kernel` (source lines 69–133) for a `bfloat16` `beta`: as
`recompute_w_u_fwd_kernel_b32`, with `beta` a `*bf16` pointer. Triton compiles the kernel for its
arguments' pointer dtypes, so lines 103–104, 115, 129 and 131 mean other operations: the products
with `b_b` are `bfloat16` products (both operands `bfloat16`), and line 131's product with the
`float32` `b_g` converts `b_kb` to `float32`. `k`, `v`, `w`, `u`, `A`, `beta` are `*bf16` pointers,
`g` `*fp32`; `cu_seqlens` and `chunk_indices` are pointers of one integer dtype `sD`
(`int32` or `int64`; the host builds `chunk_indices` with `.to(cu_seqlens)`). They are `None` on the
padded path, where the model's `IS_VARLEN = false` branch never uses them. `T`
(`do_not_specialize=['T']`: a runtime `int32` even at `1`) is `I32`. `H HV K V BT BK BV` are the
`tl.constexpr` parameters, Lean parameters; `IS_VARLEN` is a `Bool` parameter. -/
def recompute_w_u_fwd_kernel_b16 {sD : DType} (k v : Ptr .bf16) (beta : Ptr .bf16) (w u A : Ptr .bf16)
    (g : Ptr .f32) (cu_seqlens chunk_indices : Ptr sD) (T : I32)
    (H HV K V BT BK BV : Nat) (IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 91: axes 0 and 1 converted to `int64` (`.to(tl.int64)`, sign extension)
    let i_t0 : I64 := IntV.cast (tl.program_id ctx 0)
    let i_bh : I64 := IntV.cast (tl.program_id ctx 1)
    -- line 92: `i_bh // HV`, `i_bh % HV`, left to right: the constexpr `HV` built in `int64` (the
    -- tensor's dtype); truncating `//` and `%` (`tl.idiv`, `tl.irem`)
    let i_b : I64 := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    let i_h : I64 := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    -- lines 93–98: the compile-time `if IS_VARLEN`; the front end visits the taken branch only.
    -- Each branch gives the program's chunk `i_t`, the sequence's first token `bos` and the
    -- sequence length `T` (line 96 rebinds `T` to the `int64` `eos - bos`; the padded `T` is the
    -- `int32` argument, sign-extended where an `int64` operand meets it)
    let st ← (if IS_VARLEN then (do
        -- line 94: `i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)` (`i_t * 2`: the constant
        -- `2` in `int64`), then `i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)` with the
        -- old `i_t` (the right side is evaluated first; `+ 1`: the pointer plus the `int32` `1`)
        let v_n ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64))
        let i_n : I32 := (← tl.convert sD .i32 (Tensor.scalar v_n)).item
        let v_t ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
        let i_t : I64 := (← tl.convert sD .i64 (Tensor.scalar v_t)).item
        -- line 95: `bos = tl.load(cu_seqlens + i_n).to(tl.int64)`,
        -- `eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)`
        let v_b ← tl.load1 (cu_seqlens + i_n)
        let bos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_b)).item
        let v_e ← tl.load1 (cu_seqlens + i_n + (IntV.ofNat 1 : I32))
        let eos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_e)).item
        -- line 96: `T = eos - bos` (`int64`)
        let T_n : I64 := eos - bos
        pure (i_t, bos, T_n))
      else (do
        -- line 98: `bos, eos = i_b * T, i_b * T + T`: `int64` products and sum (`T` sign-extended);
        -- `eos` is not read on this path
        let bos : I64 := i_b * (IntV.cast T : I64)
        let _eos : I64 := i_b * (IntV.cast T : I64) + (IntV.cast T : I64)
        pure (i_t0, bos, (IntV.cast T : I64))) : Prog (I64 × I64 × I64))
    let (i_t, bos, T_n) := st
    -- line 99: `i_t * BT + tl.arange(0, BT)`: the constexpr in `int64`, the `int32` range
    -- sign-extended
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + ((tl.arange 0 BT).castInt : Tensor [BT] I64)
    -- line 100
    let o_A : Tensor [BT] I32 := tl.arange 0 BT
    -- line 101: the signed `int64` comparison (`T` is `int64`, or the `int32` argument
    -- sign-extended)
    let m_t : Tensor [BT] Bool := o_t.ltS T_n
    -- line 102: `m_t[:, None] & (o_A[None, :] < BT)`: the constexpr an `int32` tensor of its own
    -- type (a comparison), the broadcast to `[BT, BT]` written at once
    let m_A : Tensor [BT, BT] Bool := (m_t.col BT).and ((o_A.ltS (IntV.ofNat BT : I32)).row BT)
    -- line 103: `((beta + bos*HV) + i_h) + o_t * HV`: `int64` scalars added to the pointer, then
    -- the `int64` block (the constexpr `HV` in `int64`)
    let p_b : Tensor [BT] (Ptr .bf16) :=
      beta + bos * (IntV.ofNat HV : I64) + i_h + o_t * (IntV.ofNat HV : I64)
    -- line 104: `other=0.0`: the `float32` `+0` converted to the pointer's `bfloat16`
    let zb : Tensor [BT] BF16 ← tl.convert .f32 .bf16 (tl.full [BT] (FloatV.litF32 (FloatV.ofPat 0)))
    let b_b ← tl.load p_b (some m_t) (some zb)
    -- line 106: `((A + (bos*HV + i_h) * BT) + o_t[:, None] * (HV*BT)) + o_A[None, :]`: the `int64`
    -- scalar, then the `int64` column (`HV*BT` computed by Python, built in `int64`), then the
    -- `int32` row
    let p_A : Tensor [BT, BT] (Ptr .bf16) :=
      A + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat BT : I64) +
        o_t.col BT * (IntV.ofNat (HV * BT) : I64) + o_A.row BT
    -- line 107: `other=0.0`: the `float32` `+0` converted to the pointer's `bfloat16`
    let zA : Tensor [BT, BT] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BT] (FloatV.litF32 (FloatV.ofPat 0)))
    let b_A ← tl.load p_A (some m_A) (some zA)
    -- line 109: `range(tl.cdiv(V, BV))`: the bound is Python's `(V + (BV - 1)) // BV` (two
    -- constexprs, evaluated twice with no operation), an `int32` constant, so `i_v` is an `int32`
    -- from `0`, step `1`; the loop carries no name (every name the body assigns is first defined
    -- there and not used after the loop)
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((V + (BV - 1)) / BV) : I32) (IntV.ofInt 1 : I32) ()
        fun i_v _ => do
      -- line 110: the constexpr `BV` built in `int32`
      let o_v : Tensor [BV] I32 := i_v * (IntV.ofNat BV : I32) + tl.arange 0 BV
      -- line 111: `m_t[:, None] & (o_v[None, :] < V)`: `V` an `int32` tensor (`Inst.ok.V_lt`)
      let m_v : Tensor [BT, BV] Bool := (m_t.col BV).and ((o_v.ltS (IntV.ofNat V : I32)).row BT)
      -- line 112: as line 106, with `V` and `HV*V`
      let p_v : Tensor [BT, BV] (Ptr .bf16) :=
        v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
          o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
      -- line 113: as line 112
      let p_u : Tensor [BT, BV] (Ptr .bf16) :=
        u + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
          o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
      -- line 114: `other=0.0` as line 107
      let zV : Tensor [BT, BV] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BV] (FloatV.litF32 (FloatV.ofPat 0)))
      let b_v ← tl.load p_v (some m_v) (some zV)
      -- line 115: `b_v * b_b[:, None]`: both `bfloat16`, so the IEEE `bfloat16` product (one rounding;
      -- no addition consumes it); `.to(b_v.dtype)` is the identity on `bfloat16`
      let b_vb : Tensor [BT, BV] BF16 ← tl.mul b_v (b_b.col BV)
      -- line 116: `tl.dot` with the default accumulator (`float32` `+0`) and no addition after it
      let b_u ← tl.dot cfgU b_A b_vb (tl.zeros [BT, BV] .f32)
      -- line 117: `.to(p_u.dtype.element_ty)`: one rounding to `bfloat16`
      let s_u : Tensor [BT, BV] BF16 ← tl.cvt b_u
      tl.store p_u s_u (some m_v)
    -- line 119: `if USE_G:` is a compile-time `if`, taken (`USE_G = True`)
    -- line 120: `(g + (bos*HV + i_h)) + o_t * HV`: the parenthesised `int64` sum first
    let p_g : Tensor [BT] (Ptr .f32) :=
      g + (bos * (IntV.ofNat HV : I64) + i_h) + o_t * (IntV.ofNat HV : I64)
    -- line 121: `other=0.0` the `float32` `+0`; the helper `exp2` (lines 65–66) inlined: its
    -- `x.to(tl.float32)` is the identity on `float32`, then `tl.math.exp2`
    let l_g : Tensor [BT] F32 ← tl.load p_g (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    let b_g : Tensor [BT] F32 ← tl.exp2 l_g
    -- line 123: as line 109, with `K` and `BK`
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + (BK - 1)) / BK) : I32) (IntV.ofInt 1 : I32) ()
        fun i_k _ => do
      -- line 124
      let o_k : Tensor [BK] I32 := i_k * (IntV.ofNat BK : I32) + tl.arange 0 BK
      -- line 125: as line 111, with `K` (`Inst.ok.K_lt`)
      let m_k : Tensor [BT, BK] Bool := (m_t.col BK).and ((o_k.ltS (IntV.ofNat K : I32)).row BT)
      -- line 126: `(bos*H + i_h // (HV // H)) * K`: `HV // H` computed by Python, built in `int64`,
      -- the truncating `//` (`tl.idiv`); then as line 106 with `K` and `H*K`
      let p_k : Tensor [BT, BK] (Ptr .bf16) :=
        k + (bos * (IntV.ofNat H : I64) +
            (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
            (IntV.ofNat K : I64) +
          o_t.col BK * (IntV.ofNat (H * K) : I64) + o_k.row BT
      -- line 127: as line 106, with `K` and `HV*K`
      let p_w : Tensor [BT, BK] (Ptr .bf16) :=
        w + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) +
          o_t.col BK * (IntV.ofNat (HV * K) : I64) + o_k.row BT
      -- line 128: `other=0.0` as line 107
      let zK : Tensor [BT, BK] BF16 ← tl.convert .f32 .bf16 (tl.full [BT, BK] (FloatV.litF32 (FloatV.ofPat 0)))
      let b_k ← tl.load p_k (some m_k) (some zK)
      -- line 129: `b_k * b_b[:, None]`: both `bfloat16`, the IEEE `bfloat16` product
      let b_kb16 : Tensor [BT, BK] BF16 ← tl.mul b_k (b_b.col BK)
      -- line 130: `if USE_G:` taken; line 131: `b_kb *= b_g[:, None]` is `b_kb = b_kb * b_g[:, None]`:
      -- `bfloat16` times `float32`, so `b_kb` is converted to `float32` (exact), then the IEEE
      -- `float32` product
      let b_kbf : Tensor [BT, BK] F32 ← tl.cvt b_kb16
      let b_kb ← tl.mul b_kbf (b_g.col BK)
      -- line 132: `b_kb.to(b_k.dtype)`: one rounding to `bfloat16`; `tl.dot` with the default
      -- accumulator and no addition after it
      let b_kbh : Tensor [BT, BK] BF16 ← tl.cvt b_kb
      let b_w ← tl.dot cfgW b_A b_kbh (tl.zeros [BT, BK] .f32)
      -- line 133: one rounding to `bfloat16`
      let s_w : Tensor [BT, BK] BF16 ← tl.cvt b_w
      tl.store p_w s_w (some m_k)

/-! ## The conditions -/

/-- One launch of `recompute_w_u_fwd_kernel` as `recompute_w_u_fwd` makes it with `g` given. -/
structure Inst where
  /-- The dtype of `cu_seqlens` and `chunk_indices` (any value on the padded path, where both are
  `None`). -/
  idx : DType
  /-- `B = k.shape[0]` (`1` for a packed batch). -/
  B : Nat
  /-- `T = k.shape[1]`, the runtime scalar `T` (the packed batch's total tokens). -/
  T : Nat
  /-- `H = k.shape[2]`, the key heads. -/
  H : Nat
  /-- `HV = v.shape[2]`, the value heads. -/
  HV : Nat
  /-- `K = k.shape[3]`. -/
  K : Nat
  /-- `V = v.shape[3]`. -/
  V : Nat
  /-- `N`, the packed sequences (`cu_seqlens.shape[0] - 1`; unused when padded). -/
  N : Nat
  /-- `NT = len(chunk_indices)`, the grid's axis 0 when packed (unused when padded). -/
  NT : Nat
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  isVarlen : Bool
  /-- `k`: contiguous `bfloat16` `[B, T, H, K]`. -/
  k : Buf .bf16
  /-- `v`: contiguous `bfloat16` `[B, T, HV, V]`. -/
  v : Buf .bf16
  /-- `beta.dtype == torch.bfloat16` (`false`: `float32`). -/
  bbf : Bool
  /-- `beta` when `float32` (`bbf = false`): contiguous `[B, T, HV]` (an empty buffer otherwise). -/
  beta : Buf .f32
  /-- `beta` when `bfloat16` (`bbf = true`): contiguous `[B, T, HV]` (an empty buffer otherwise). -/
  betab : Buf .bf16
  /-- `A`: contiguous `bfloat16` `[B, T, HV, 64]`. -/
  A : Buf .bf16
  /-- `g`: contiguous `float32` `[B, T, HV]`. -/
  g : Buf .f32
  /-- `w`: the fresh output `[B, T, HV, K]`, `bfloat16`. -/
  w : Buf .bf16
  /-- `u`: the fresh output `[B, T, HV, V]`, `bfloat16`. -/
  u : Buf .bf16
  /-- `cu_seqlens`: `[N + 1]` (an empty buffer when padded). -/
  cu : Buf idx
  /-- `chunk_indices`: `[NT, 2]` (an empty buffer when padded). -/
  ci : Buf idx

/-- The tokens of the batch, `B · T` (`T` when packed, with `B = 1`). -/
def nTok (d : Inst) : Nat := d.B * d.T

/-- The grid's axis 0 (source line 191): `len(chunk_indices)` packed, `cdiv(T, 64)` padded. -/
def g0 (d : Inst) : Nat := if d.isVarlen then d.NT else (d.T + 63) / 64

/-- The instances covered. Every field is either the host's own requirement, a bound outside the
setup's real use, or a limit of the kernel (README, Exclusions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `cu_seqlens` (and `chunk_indices`) are `int32` or `int64` (SETUP.md). -/
  idx_ok : d.idx = .i32 ∨ d.idx = .i64
  /-- At least one key head. -/
  h_pos : 1 ≤ d.H
  /-- At least one value head. -/
  hv_pos : 1 ≤ d.HV
  /-- `HV` is a multiple of `H` (grouped value attention; `chunk_gated_delta_rule` checks it,
  `fla/ops/gated_delta_rule/chunk.py:526-530`). -/
  hv_div : d.HV % d.H = 0
  /-- `K ≥ 1`. -/
  k_pos : 1 ≤ d.K
  /-- `V ≥ 1`. -/
  v_pos : 1 ≤ d.V
  /-- `K + 64 ≤ 2^31`: the column offsets `o_k` (line 124) and the comparison with `K` (line 125)
  are `int32` (beyond: head dims of `2^31`, outside real use). -/
  K_lt : d.K + 64 ≤ 2 ^ 31
  /-- `V + 64 ≤ 2^31`: as `K_lt`, lines 110–111. -/
  V_lt : d.V + 64 ≤ 2 ^ 31
  /-- `T < 2^31`: `T` is an `int32` argument (beyond: `2^31` tokens, outside real use). -/
  t_lt : d.T < 2 ^ 31
  /-- `T` is a context length (the sequence length, or the packed batch's total tokens): at most
  `2^22`. -/
  t_ctx : d.T ≤ 2 ^ 22
  /-- `B` is a batch size (the padded batch's sequences; `1` when packed): at most `8192`. -/
  b_batch : d.B ≤ 8192
  /-- `N` is a batch size (the packed batch's sequences): at most `8192`. -/
  n_batch : d.isVarlen = true → d.N ≤ 8192
  /-- The element offsets, masked lanes included, stay far inside `int64` (tensors far below
  `2^62` elements). -/
  size_lt : (nTok d + 64) * d.HV * (d.K + d.V + 64) < 2 ^ 62
  /-- `k` holds `[B, T, H, K]`. -/
  k_len : nTok d * d.H * d.K ≤ d.k.len
  /-- `v` holds `[B, T, HV, V]`. -/
  v_len : nTok d * d.HV * d.V ≤ d.v.len
  /-- `beta` holds `[B, T, HV]` (`float32`). -/
  beta_len : d.bbf = false → nTok d * d.HV ≤ d.beta.len
  /-- `beta` holds `[B, T, HV]` (`bfloat16`). -/
  betab_len : d.bbf = true → nTok d * d.HV ≤ d.betab.len
  /-- `A` holds `[B, T, HV, 64]`. -/
  A_len : nTok d * d.HV * 64 ≤ d.A.len
  /-- `g` holds `[B, T, HV]`. -/
  g_len : nTok d * d.HV ≤ d.g.len
  /-- `w` holds `[B, T, HV, K]`. -/
  w_len : nTok d * d.HV * d.K ≤ d.w.len
  /-- `u` holds `[B, T, HV, V]`. -/
  u_len : nTok d * d.HV * d.V ≤ d.u.len
  /-- A packed batch is `[1, T, ·, ·]` (`chunk_gated_delta_rule` requires it with `cu_seqlens`,
  `chunk.py:551-553`). -/
  varlen_b : d.isVarlen = true → d.B = 1
  /-- `cu_seqlens` holds `N + 1` entries. -/
  cu_len : d.isVarlen = true → d.N + 1 ≤ d.cu.len
  /-- `chunk_indices` holds `[NT, 2]`. -/
  ci_len : d.isVarlen = true → 2 * d.NT ≤ d.ci.len
  /-- **Kernel limit**: fewer than `2^31` sequences (line 94 truncates the sequence index to
  `int32`). Not reached by real use (README, Exclusions). -/
  n_lt : d.isVarlen = true → d.N < 2 ^ 31
  /-- The grid's axis 0 is below CUDA's `2^31`. -/
  grid_t : g0 d < 2 ^ 31
  /-- **Kernel limit**: the grid's axis 1 is `B · HV`, and CUDA allows at most `65535`: a padded
  batch with `B · HV ≥ 65536` fails to launch (README, Exclusions). -/
  grid_bh : d.B * d.HV < 2 ^ 16
  /-- Fewer than `2^31` programs (Triton's launcher). -/
  grid_size : g0 d * (d.B * d.HV) < 2 ^ 31
  /-- `w` is fresh (`k.new_empty`): it shares no byte with an input or with `u`. -/
  w_k : Disjoint d.w.bytes d.k.bytes
  w_v : Disjoint d.w.bytes d.v.bytes
  w_beta : Disjoint d.w.bytes d.beta.bytes
  w_betab : Disjoint d.w.bytes d.betab.bytes
  w_A : Disjoint d.w.bytes d.A.bytes
  w_g : Disjoint d.w.bytes d.g.bytes
  w_cu : Disjoint d.w.bytes d.cu.bytes
  w_ci : Disjoint d.w.bytes d.ci.bytes
  w_u : Disjoint d.w.bytes d.u.bytes
  /-- `u` is fresh (`torch.empty_like(v)`): it shares no byte with an input. -/
  u_k : Disjoint d.u.bytes d.k.bytes
  u_v : Disjoint d.u.bytes d.v.bytes
  u_beta : Disjoint d.u.bytes d.beta.bytes
  u_betab : Disjoint d.u.bytes d.betab.bytes
  u_A : Disjoint d.u.bytes d.A.bytes
  u_g : Disjoint d.u.bytes d.g.bytes
  u_cu : Disjoint d.u.bytes d.cu.bytes
  u_ci : Disjoint d.u.bytes d.ci.bytes

/-- The model on an instance: the kernel compiled for `beta`'s dtype, the buffers' first elements,
`T` as an `int32`, the shapes, and the host's block sizes `BT = A.shape[-1] = 64`, `BK = BV = 64`. -/
def model (d : Inst) : Kernel :=
  if d.bbf then
    recompute_w_u_fwd_kernel_b16 (sD := d.idx) (d.k.ptr 0) (d.v.ptr 0) (d.betab.ptr 0) (d.w.ptr 0)
      (d.u.ptr 0) (d.A.ptr 0) (d.g.ptr 0) (d.cu.ptr 0) (d.ci.ptr 0) (IntV.ofNat d.T)
      d.H d.HV d.K d.V 64 64 64 d.isVarlen
  else
    recompute_w_u_fwd_kernel_b32 (sD := d.idx) (d.k.ptr 0) (d.v.ptr 0) (d.beta.ptr 0) (d.w.ptr 0)
      (d.u.ptr 0) (d.A.ptr 0) (d.g.ptr 0) (d.cu.ptr 0) (d.ci.ptr 0) (IntV.ofNat d.T)
      d.H d.HV d.K d.V 64 64 64 d.isVarlen

/-- The source's block shapes: `[64]` (`o_t`, `o_A`, `m_t`, `p_b`, `b_b`, `o_v`, `o_k`, `p_g`,
`b_g`), `[64, 1]` (`m_t[:, None]`, `o_t[:, None]`, `b_b[:, None]`, `b_g[:, None]`), `[1, 64]`
(`o_A[None, :]`, `o_v[None, :]`, `o_k[None, :]`) and `[64, 64]` (every tile). -/
def blockShapes : List Shape := [[64], [64, 1], [1, 64], [64, 64]]

/-- The host's launch (lines 197–219): the grid `(NT, B·HV)`; `enable_fp_fusion` at Triton's default
(`true`); `num_warps = 4`, `num_stages = 3` change no meaning; `num_ctas = 1`, no `launch_pdl`,
`debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![g0 d, d.B * d.HV, 1], fusion := true, blocks := blockShapes }

/-- The integer an index element holds (`int32` or `int64`). -/
def idxVal : (d : DType) → d.Val → Int
  | .i32, v => v.toInt
  | .i64, v => v.toInt
  | _, _ => 0

/-- Element `j` of an index buffer, as an integer, when present. -/
def idxAt {dt : DType} (b : Buf dt) (m : Mem) (j : Nat) : Option Int :=
  (m.read dt (b.ptr (j : Int)).addr).map (idxVal dt)

/-- Element `j` of a `bfloat16` buffer. -/
def bfAt (b : Buf .bf16) (m : Mem) (j : Nat) : Option BF16 := m.read .bf16 (b.ptr (j : Int)).addr

/-- Element `j` of a `float32` buffer. -/
def f32At (b : Buf .f32) (m : Mem) (j : Nat) : Option F32 := m.read .f32 (b.ptr (j : Int)).addr

/-- Element `j` of a `bfloat16` buffer, `+0` when absent (`Pre` makes every element read present). -/
def bfV (b : Buf .bf16) (m : Mem) (j : Nat) : BF16 := (bfAt b m j).getD (FloatV.ofPat 0)

/-- Element `j` of a `float32` buffer, `+0` when absent. -/
def f32V (b : Buf .f32) (m : Mem) (j : Nat) : F32 := (f32At b m j).getD (FloatV.ofPat 0)

/-- `cu_seqlens[n]` (as a natural number). -/
def cuN (d : Inst) (m : Mem) (n : Nat) : Nat := ((idxAt d.cu m n).getD 0).toNat

/-- The length of sequence `n`, `cu_seqlens[n + 1] - cu_seqlens[n]`. -/
def seqLen (d : Inst) (m : Mem) (n : Nat) : Nat := cuN d m (n + 1) - cuN d m n

/-- `chunk_indices[i, 0]`, the sequence of program `i`. -/
def chSeq (d : Inst) (m : Mem) (i : Nat) : Nat := ((idxAt d.ci m (2 * i)).getD 0).toNat

/-- `chunk_indices[i, 1]`, the chunk of program `i` in its sequence. -/
def chChunk (d : Inst) (m : Mem) (i : Nat) : Nat := ((idxAt d.ci m (2 * i + 1)).getD 0).toNat

/-- The packed batch's metadata, as fla passes it: `cu_seqlens` non-decreasing from `0` to `T`, and
`chunk_indices` listing every (sequence, chunk of 64 tokens) pair exactly once, as
`prepare_chunk_indices` builds it (lines 164–172; any order). -/
structure SeqOk (d : Inst) (init : Mem) : Prop where
  cuv : d.cu.Valid init
  civ : d.ci.Valid init
  cu_some : ∀ n, n ≤ d.N → ∃ x, idxAt d.cu init n = some x ∧ 0 ≤ x
  cu_zero : cuN d init 0 = 0
  cu_last : cuN d init d.N = d.T
  cu_mono : ∀ n, n < d.N → cuN d init n ≤ cuN d init (n + 1)
  ci_some : ∀ i, i < 2 * d.NT → ∃ x, idxAt d.ci init i = some x ∧ 0 ≤ x
  ci_ok : ∀ i, i < d.NT → chSeq d init i < d.N ∧ chChunk d init i * 64 < seqLen d init (chSeq d init i)
  ci_all : ∀ n j, n < d.N → j * 64 < seqLen d init n →
    ∃ i, i < d.NT ∧ chSeq d init i = n ∧ chChunk d init i = j
  ci_once : ∀ i i', i < d.NT → i' < d.NT → chSeq d init i = chSeq d init i' →
    chChunk d init i = chChunk d init i' → i = i'

/-- Element `(f, h, c)` of `k` (`[·, H, K]`), `f` the flat token (`b · T + t`). -/
def kIdx (d : Inst) (f h c : Nat) : Nat := (f * d.H + h) * d.K + c

/-- Element `(f, h, c)` of `v` and `u` (`[·, HV, V]`). -/
def vIdx (d : Inst) (f h c : Nat) : Nat := (f * d.HV + h) * d.V + c

/-- Element `(f, h, c)` of `w` (`[·, HV, K]`). -/
def wIdx (d : Inst) (f h c : Nat) : Nat := (f * d.HV + h) * d.K + c

/-- Element `(f, h)` of `beta` and `g` (`[·, HV]`). -/
def hIdx (d : Inst) (f h : Nat) : Nat := f * d.HV + h

/-- Element `(f, h, j)` of `A` (`[·, HV, 64]`). -/
def aIdx (d : Inst) (f h j : Nat) : Nat := (f * d.HV + h) * 64 + j

/-- A `k`, `v` value the claim covers: finite, `|x| ≤ 2^32`. -/
def KVOk (x : BF16) : Prop := x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 32

/-- An `A` value the claim covers: finite, `|x| ≤ 2^64`. -/
def AOk (x : BF16) : Prop := x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 64

/-- A `float32` `beta` value the claim covers: finite, `|x| ≤ 2^8`. -/
def BetaOk (x : F32) : Prop := x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 8

/-- A `bfloat16` `beta` value the claim covers: finite, `|x| ≤ 2^8`. -/
def BetaOkB (x : BF16) : Prop := x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 8

/-- A `g` value the claim covers: finite, `x ≤ 8` (any negative value). -/
def GOk (x : F32) : Prop := x.isFinite = true ∧ x.toRat ≤ 8

/-- The inputs covered: the buffers valid, every element of the inputs' tensors present and in
range (README, Exclusions: real values are far inside), and, packed, the metadata as fla builds
it. -/
structure Pre (d : Inst) (init : Mem) : Prop where
  kv : d.k.Valid init
  vv : d.v.Valid init
  betav : d.bbf = false → d.beta.Valid init
  betabv : d.bbf = true → d.betab.Valid init
  Av : d.A.Valid init
  gv : d.g.Valid init
  wv : d.w.Valid init
  uv : d.u.Valid init
  k_vals : ∀ f h c, f < nTok d → h < d.H → c < d.K → ∃ x, bfAt d.k init (kIdx d f h c) = some x ∧ KVOk x
  v_vals : ∀ f h c, f < nTok d → h < d.HV → c < d.V → ∃ x, bfAt d.v init (vIdx d f h c) = some x ∧ KVOk x
  beta_vals : d.bbf = false → ∀ f h, f < nTok d → h < d.HV →
    ∃ x, f32At d.beta init (hIdx d f h) = some x ∧ BetaOk x
  betab_vals : d.bbf = true → ∀ f h, f < nTok d → h < d.HV →
    ∃ x, bfAt d.betab init (hIdx d f h) = some x ∧ BetaOkB x
  g_vals : ∀ f h, f < nTok d → h < d.HV → ∃ x, f32At d.g init (hIdx d f h) = some x ∧ GOk x
  A_vals : ∀ f h j, f < nTok d → h < d.HV → j < 64 → ∃ x, bfAt d.A init (aIdx d f h j) = some x ∧ AOk x
  seqs : d.isVarlen = true → SeqOk d init

/-! ## The specification

A sequence is its first flat token `s0` and its length `len`: padded, `(b · T, T)` for `b < B`;
packed, `(cu_seqlens[n], cu_seqlens[n+1] - cu_seqlens[n])` for `n < N`. Token `t < len` of it is the
flat token `s0 + t`; its chunk starts at `t / 64 · 64`, and column `j < 64` of its row of `A` belongs
to the chunk's token `t / 64 · 64 + j`, which is inside the sequence when `t / 64 · 64 + j < len`. -/

/-- Whether column `j` of token `t`'s row is a token of the sequence (of length `len`). -/
def inSeq (len t j : Nat) : Bool := decide (t / 64 * 64 + j < len)

/-- The flat token of column `j` of token `t`'s row, in the sequence starting at `s0`. -/
def colTok (s0 t j : Nat) : Nat := s0 + t / 64 * 64 + j

/-- `A`'s entry in column `j` of flat token `f`'s row, value head `h`. -/
def aV (d : Inst) (m : Mem) (f h j : Nat) : BF16 := bfV d.A m (aIdx d f h j)

/-- Line 115's operand `b_vb` for `v[f, h, c]`: with a `float32` `beta`, the `bfloat16` rounding of
the `float32` product `v[f, h, c] · beta[f, h]`; with a `bfloat16` `beta`, the `bfloat16` product. -/
def vbV (d : Inst) (m : Mem) (f h c : Nat) : BF16 :=
  if d.bbf then FloatV.mul (bfV d.v m (vIdx d f h c)) (bfV d.betab m (hIdx d f h))
  else FloatV.cvt (FloatV.mul (FloatV.cvt (bfV d.v m (vIdx d f h c)) : F32) (f32V d.beta m (hIdx d f h)))

/-- The reference of `u[s0 + t, h, c]`: `Σ_j A[s0+t, h, j] · vb[s0 + t/64·64 + j, h, c]` over the
columns inside the sequence. -/
def uRef (d : Inst) (m : Mem) (s0 len t h c : Nat) : ℚ :=
  ∑ j ∈ Finset.range 64, if inSeq len t j then
    (aV d m (s0 + t) h j).toRat * (vbV d m (colTok s0 t j) h c).toRat else 0

/-- The products' lifted magnitudes (`Rules.Num.liftMag .bf16`, a subnormal counted at `2^-126`),
`Σ_j liftMag A · liftMag vb`: the `L` of `Rules.Num.DotBF16` for line 116's dot. -/
def uMag (d : Inst) (m : Mem) (s0 len t h c : Nat) : ℚ :=
  ∑ j ∈ Finset.range 64, if inSeq len t j then
    Rules.Num.liftMag .bf16 (aV d m (s0 + t) h j).toRat *
      Rules.Num.liftMag .bf16 (vbV d m (colTok s0 t j) h c).toRat else 0

/-- The tolerance of `u`: one `bfloat16` rounding of the result (`2^-8·|ref|`), the dot product's
error relative to the lifted magnitude `L` (`2^-15·L`), and an absolute floor (`2^-119`) (README,
Tolerances). -/
def tolU (ref L : ℚ) : ℚ :=
  2 ^ (-8 : ℤ) * |ref| + 2 ^ (-15 : ℤ) * L + 2 ^ (-119 : ℤ)

/-- `u[s0 + t, h, c]` in the final memory: a finite `bfloat16` within `tolU` of `uRef`. -/
def CellU (d : Inst) (init fin : Mem) (s0 len t h c : Nat) : Prop :=
  ∃ o : BF16, bfAt d.u fin (vIdx d (s0 + t) h c) = some o ∧ o.isFinite = true ∧
    |o.toRat - uRef d init s0 len t h c| ≤ tolU (uRef d init s0 len t h c) (uMag d init s0 len t h c)

/-- The key head that value head `h` reads: `h / (HV / H)`. -/
def kHead (d : Inst) (h : Nat) : Nat := h / (d.HV / d.H)

/-- `beta[f, h]`, exactly (from the buffer of `beta`'s dtype). -/
noncomputable def betaR (d : Inst) (m : Mem) (f h : Nat) : ℝ :=
  if d.bbf then (bfV d.betab m (hIdx d f h)).toReal else (f32V d.beta m (hIdx d f h)).toReal

/-- `k[f, h / (HV/H), c] · beta[f, h]`, exactly. -/
noncomputable def kbX (d : Inst) (m : Mem) (f h c : Nat) : ℝ :=
  (bfV d.k m (kIdx d f (kHead d h) c)).toReal * betaR d m f h

/-- `2^(g[f, h])`, exactly. -/
noncomputable def gE (d : Inst) (m : Mem) (f h : Nat) : ℝ := (2 : ℝ) ^ (f32V d.g m (hIdx d f h)).toReal

/-- The reference of `w[s0 + t, h, c]`: `Σ_j A[s0+t, h, j] · k[f_j, ·, c] · beta[f_j, h] · 2^g[f_j, h]`
over the columns inside the sequence (`f_j = s0 + t/64·64 + j`), exactly. -/
noncomputable def wRef (d : Inst) (m : Mem) (s0 len t h c : Nat) : ℝ :=
  ∑ j ∈ Finset.range 64, if inSeq len t j then
    (aV d m (s0 + t) h j).toReal * kbX d m (colTok s0 t j) h c * gE d m (colTok s0 t j) h else 0

/-- `Σ_j liftMag A · |k·beta| · 2^g`: the scale of the per-term roundings (line 132) and of the
dot product's error. -/
noncomputable def wMag (d : Inst) (m : Mem) (s0 len t h c : Nat) : ℝ :=
  ∑ j ∈ Finset.range 64, if inSeq len t j then
    (Rules.Num.liftMag .bf16 (aV d m (s0 + t) h j).toRat : ℝ) * |kbX d m (colTok s0 t j) h c| *
      gE d m (colTok s0 t j) h else 0

/-- `Σ_j liftMag A · (|k·beta| + 1)`: the scale of the absolute floors (the `2^-126` of
`tl.math.exp2`, subnormal roundings). -/
noncomputable def wFloor (d : Inst) (m : Mem) (s0 len t h c : Nat) : ℝ :=
  ∑ j ∈ Finset.range 64, if inSeq len t j then
    (Rules.Num.liftMag .bf16 (aV d m (s0 + t) h j).toRat : ℝ) * (|kbX d m (colTok s0 t j) h c| + 1)
    else 0

/-- The tolerance of `w`: `2^-6` of the magnitude `M = wMag` (the rounding of the result, at most
`2^-8·|ref| ≤ 2^-8·M`, each operand's one or two roundings and the dot product's error), the
absolute floors' `2^-124·F` (`F = wFloor`) and `2^-119` (README, Tolerances). -/
noncomputable def tolW (M F : ℝ) : ℝ :=
  2 ^ (-6 : ℤ) * M + 2 ^ (-124 : ℤ) * F + 2 ^ (-119 : ℤ)

/-- `w[s0 + t, h, c]` in the final memory: a finite `bfloat16` within `tolW` of `wRef`. -/
def CellW (d : Inst) (init fin : Mem) (s0 len t h c : Nat) : Prop :=
  ∃ o : BF16, bfAt d.w fin (wIdx d (s0 + t) h c) = some o ∧ o.isFinite = true ∧
    |o.toReal - wRef d init s0 len t h c| ≤
      tolW (wMag d init s0 len t h c) (wFloor d init s0 len t h c)

/-- Every column of `u` and `w` of token `t < len` of the sequence at `s0`, head `h`. -/
def TokOk (d : Inst) (init fin : Mem) (s0 len t h : Nat) : Prop :=
  (∀ c, c < d.V → CellU d init fin s0 len t h c) ∧ (∀ c, c < d.K → CellW d init fin s0 len t h c)

/-- The elements of `u`'s storage that make up `u`. -/
def uSet (d : Inst) : Set Nat := {j | j < nTok d * d.HV * d.V}

/-- The elements of `w`'s storage that make up `w`. -/
def wSet (d : Inst) : Set Nat := {j | j < nTok d * d.HV * d.K}

/-- What the launch leaves. Padded: sequence `b < B` is the flat tokens `b·T … b·T + T − 1`. Packed:
sequence `n < N` is `cu_seqlens[n] … cu_seqlens[n+1] − 1`. Every token of every sequence, every value
head, meets `TokOk`; every byte outside `u` and `w` is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (d.isVarlen = false → ∀ b t h, b < d.B → t < d.T → h < d.HV →
      TokOk d init fin (b * d.T) d.T t h) ∧
  (d.isVarlen = true → ∀ n t h, n < d.N → t < seqLen d init n → h < d.HV →
      TokOk d init fin (cuN d init n) (seqLen d init n) t h) ∧
  ∀ z, z ∉ d.u.elems (uSet d) → z ∉ d.w.elems (wSet d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32` (lines 116, 132). -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotTotal`: line 116's `tl.dot` has a result for all operands. -/
  dot_total_u : Rules.Num.DotTotal P cfgU
  /-- [rule] `Num.DotTotal`: line 132's `tl.dot` has a result for all operands. -/
  dot_total_w : Rules.Num.DotTotal P cfgW
  /-- [rule] `Num.Exp2F32`: `tl.math.exp2` on `float32` (line 121, through the helper `exp2`). -/
  exp2 : Rules.Num.Exp2F32 P
  /-- [rule] `Num.MathTotal`: `tl.math.exp2` on `float32` has a result for every input. -/
  exp2_total : Rules.Num.MathTotal P .exp2 .binary32

/-- The claim to prove. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16,
    Platform.ref.dotTotal _, Platform.ref.dotTotal _, Platform.sm90ref.exp2F32,
    Platform.ref.mathTotal⟩⟩

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d₁ d₂ : DType} {b : Buf d₁} {c : Buf d₂}
    (h : c.base + c.len * d₂.bytes ≤ b.base ∨ b.base + b.len * d₁.bytes ≤ c.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro z hb hc
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hb hc
  omega

/-- A memory in which every byte is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

theorem zeroMem_valid {dt : DType} {b : Buf dt} (h1 : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (h3 : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨h1, h2, h3, fun _ _ => by simp [zeroMem]⟩

theorem zeroMem_read_bf16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    zeroMem.read .bf16 a = some (FloatV.ofPat 0 : BF16) := by
  have hb : zeroMem.readBytes a DType.bf16.bytes = some [0, 0] := by
    show zeroMem.readBytes a 2 = some [0, 0]
    simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

theorem zeroMem_read_f32 {a : Int} (ha : Mem.Aligned .f32 a) :
    zeroMem.read .f32 a = some (FloatV.ofPat 0 : F32) := by
  have hb : zeroMem.readBytes a DType.f32.bytes = some [0, 0, 0, 0] := by
    show zeroMem.readBytes a 4 = some [0, 0, 0, 0]
    simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

theorem zero_toRat_bf16 : (FloatV.ofPat 0 : BF16).toRat = 0 := by decide +kernel

theorem zero_toRat_f32 : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel

theorem zero_kvok : KVOk (FloatV.ofPat 0 : BF16) :=
  ⟨by decide +kernel, by rw [zero_toRat_bf16]; norm_num⟩

theorem zero_aok : AOk (FloatV.ofPat 0 : BF16) :=
  ⟨by decide +kernel, by rw [zero_toRat_bf16]; norm_num⟩

theorem zero_betaok : BetaOk (FloatV.ofPat 0 : F32) :=
  ⟨by decide +kernel, by rw [zero_toRat_f32]; norm_num⟩

theorem zero_gok : GOk (FloatV.ofPat 0 : F32) :=
  ⟨by decide +kernel, by rw [zero_toRat_f32]; norm_num⟩

theorem zeroMem_bfAt {b : Buf .bf16} (hv : b.Valid zeroMem) {j : Nat} (hj : j < b.len) :
    bfAt b zeroMem j = some (FloatV.ofPat 0 : BF16) :=
  zeroMem_read_bf16 (Buf.aligned b hv hj)

theorem zeroMem_f32At {b : Buf .f32} (hv : b.Valid zeroMem) {j : Nat} (hj : j < b.len) :
    f32At b zeroMem j = some (FloatV.ofPat 0 : F32) :=
  zeroMem_read_f32 (Buf.aligned b hv hj)

/-! ## Coverage of the setup -/

/-- The host's launch for these sizes: contiguous tensors, the ten storages `2^40` bytes apart;
`cu_seqlens` and `chunk_indices` empty when padded, and the `beta` buffer of the other dtype
empty. -/
def canon (idx : DType) (bbf : Bool) (B T H HV K V N NT : Nat) (varlen : Bool) : Inst where
  idx := idx
  bbf := bbf
  B := B
  T := T
  H := H
  HV := HV
  K := K
  V := V
  N := N
  NT := NT
  isVarlen := varlen
  k := ⟨2 ^ 40, B * T * H * K⟩
  v := ⟨2 * 2 ^ 40, B * T * HV * V⟩
  beta := ⟨3 * 2 ^ 40, if bbf then 0 else B * T * HV⟩
  betab := ⟨10 * 2 ^ 40, if bbf then B * T * HV else 0⟩
  A := ⟨4 * 2 ^ 40, B * T * HV * 64⟩
  g := ⟨5 * 2 ^ 40, B * T * HV⟩
  w := ⟨6 * 2 ^ 40, B * T * HV * K⟩
  u := ⟨7 * 2 ^ 40, B * T * HV * V⟩
  cu := ⟨8 * 2 ^ 40, if varlen then N + 1 else 0⟩
  ci := ⟨9 * 2 ^ 40, if varlen then 2 * NT else 0⟩

/-! Each `setup_*_ok` proves every field of `Inst.ok`: the arithmetic by `decide`, the
disjointness from the storages' byte ranges (`bytes_disj`). -/

/-- `tests/ops/test_gdn.py:157, 162` (`test_chunk`), in `float16` there, here in `bfloat16`: `B = 4`, `T = 1024`, `H = 4`, `HV = 4`, `K = 128`, `V = 128`, padded, `float32` `beta`; `NT = 16`. -/
def setup_chunk_b4_t1024_h4_d128 : Inst := canon .i64 false 4 1024 4 4 128 128 4 16 false

theorem setup_chunk_b4_t1024_h4_d128_ok : setup_chunk_b4_t1024_h4_d128.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:158`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 75`, `H = 4`, `HV = 4`, `K = 64`, `V = 64`, padded, `float32` `beta`; `NT = 2`. -/
def setup_chunk_b2_t75_h4_d64 : Inst := canon .i64 false 2 75 4 4 64 64 2 2 false

theorem setup_chunk_b2_t75_h4_d64_ok : setup_chunk_b2_t75_h4_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:159`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 500`, `H = 3`, `HV = 3`, `K = 60`, `V = 60`, padded, `float32` `beta`; `NT = 8`. -/
def setup_chunk_b2_t500_h3_d60 : Inst := canon .i64 false 2 500 3 3 60 60 2 8 false

theorem setup_chunk_b2_t500_h3_d60_ok : setup_chunk_b2_t500_h3_d60.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:160`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 1000`, `H = 3`, `HV = 3`, `K = 64`, `V = 64`, padded, `float32` `beta`; `NT = 16`. -/
def setup_chunk_b2_t1000_h3_d64 : Inst := canon .i64 false 2 1000 3 3 64 64 2 16 false

theorem setup_chunk_b2_t1000_h3_d64_ok : setup_chunk_b2_t1000_h3_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:161`, in `float16` there, here in `bfloat16`: `B = 3`, `T = 1024`, `H = 4`, `HV = 4`, `K = 100`, `V = 100`, padded, `float32` `beta`; `NT = 16`. -/
def setup_chunk_b3_t1024_h4_d100 : Inst := canon .i64 false 3 1024 4 4 100 100 3 16 false

theorem setup_chunk_b3_t1024_h4_d100_ok : setup_chunk_b3_t1024_h4_d100.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:163`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 1500`, `H = 4`, `HV = 4`, `K = 128`, `V = 128`, padded, `float32` `beta`; `NT = 24`. -/
def setup_chunk_b2_t1500_h4_d128 : Inst := canon .i64 false 2 1500 4 4 128 128 2 24 false

theorem setup_chunk_b2_t1500_h4_d128_ok : setup_chunk_b2_t1500_h4_d128.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:164`, in `float16` there, here in `bfloat16`: `B = 4`, `T = 2048`, `H = 8`, `HV = 8`, `K = 64`, `V = 64`, padded, `float32` `beta`; `NT = 32`. -/
def setup_chunk_b4_t2048_h8_d64 : Inst := canon .i64 false 4 2048 8 8 64 64 4 32 false

theorem setup_chunk_b4_t2048_h8_d64_ok : setup_chunk_b4_t2048_h8_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:165`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 256`, `H = 2`, `HV = 4`, `K = 64`, `V = 64`, padded, `float32` `beta`; `NT = 4`. -/
def setup_chunk_b2_t256_h2_hv4_d64 : Inst := canon .i64 false 2 256 2 4 64 64 2 4 false

theorem setup_chunk_b2_t256_h2_hv4_d64_ok : setup_chunk_b2_t256_h2_hv4_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:166`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 512`, `H = 2`, `HV = 8`, `K = 64`, `V = 64`, padded, `float32` `beta`; `NT = 8`. -/
def setup_chunk_b2_t512_h2_hv8_d64 : Inst := canon .i64 false 2 512 2 8 64 64 2 8 false

theorem setup_chunk_b2_t512_h2_hv8_d64_ok : setup_chunk_b2_t512_h2_hv8_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:167`, in `float16` there, here in `bfloat16`: `B = 2`, `T = 1024`, `H = 4`, `HV = 8`, `K = 128`, `V = 128`, padded, `float32` `beta`; `NT = 16`. -/
def setup_chunk_b2_t1024_h4_hv8_d128 : Inst := canon .i64 false 2 1024 4 8 128 128 2 16 false

theorem setup_chunk_b2_t1024_h4_hv8_d128_ok : setup_chunk_b2_t1024_h4_hv8_d128.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:165, 167` (`test_recompute_w_u_fwd`; the test's `bfloat16` `beta`, line 185): `B = 2`, `T = 128`, `H = 2`, `HV = 2`, `K = 64`, `V = 64`, padded, `bfloat16` `beta`; `NT = 2`. -/
def setup_wu_b2_t128_h2_d64 : Inst := canon .i64 true 2 128 2 2 64 64 2 2 false

theorem setup_wu_b2_t128_h2_d64_ok : setup_wu_b2_t128_h2_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:166`: `B = 2`, `T = 128`, `H = 2`, `HV = 4`, `K = 64`, `V = 64`, padded, `bfloat16` `beta`; `NT = 2`. -/
def setup_wu_b2_t128_h2_hv4_d64 : Inst := canon .i64 true 2 128 2 4 64 64 2 2 false

theorem setup_wu_b2_t128_h2_hv4_d64_ok : setup_wu_b2_t128_h2_hv4_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:168` (run in `float16` there): `B = 1`, `T = 256`, `H = 4`, `HV = 4`, `K = 32`, `V = 32`, padded, `bfloat16` `beta`; `NT = 4`. -/
def setup_wu_b1_t256_h4_d32 : Inst := canon .i64 true 1 256 4 4 32 32 1 4 false

theorem setup_wu_b1_t256_h4_d32_ok : setup_wu_b1_t256_h4_d32.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- Qwen3-Next's gated DeltaNet heads (16 key, 32 value heads of 128): `B = 2`, `T = 4096`, `H = 16`, `HV = 32`, `K = 128`, `V = 128`, padded, `float32` `beta`; `NT = 64`. -/
def setup_qwen3next_b2_t4096 : Inst := canon .i64 false 2 4096 16 32 128 128 2 64 false

theorem setup_qwen3next_b2_t4096_ok : setup_qwen3next_b2_t4096.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- fla's `GatedDeltaNet` defaults (`fla/layers/gated_deltanet.py:93-96`: 6 heads, head dim 256, `expand_v = 2`): `B = 2`, `T = 4096`, `H = 6`, `HV = 6`, `K = 256`, `V = 512`, padded, `float32` `beta`; `NT = 64`. -/
def setup_fla_gdn_b2_t4096 : Inst := canon .i64 false 2 4096 6 6 256 512 2 64 false

theorem setup_fla_gdn_b2_t4096_ok : setup_fla_gdn_b2_t4096.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- the example of `chunk_gated_delta_rule`'s docstring (`fla/ops/gated_delta_rule/chunk.py:487, 491`: `bfloat16` `beta`): `B = 4`, `T = 2048`, `H = 4`, `HV = 8`, `K = 512`, `V = 512`, padded, `bfloat16` `beta`; `NT = 32`. -/
def setup_doc_b4_t2048_h4_hv8_d512 : Inst := canon .i64 true 4 2048 4 8 512 512 4 32 false

theorem setup_doc_b4_t2048_h4_hv8_d512_ok : setup_doc_b4_t2048_h4_hv8_d512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:758` (`test_chunk_varlen`), in `float16` there, here in `bfloat16`: `B = 1`, `T = 15`, `H = 4`, `HV = 4`, `K = 60`, `V = 60`, packed, `cu_seqlens = [0, 15]`, `i64`, `float32` `beta`; `NT = 1`. -/
def setup_varlen_h4_d60_n1 : Inst := canon .i64 false 1 15 4 4 60 60 1 1 true

theorem setup_varlen_h4_d60_n1_ok : setup_varlen_h4_d60_n1.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:759-760`, in `float16` there, here in `bfloat16`: `B = 1`, `T = 1000`, `H = 4`, `HV = 4`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 256, 500, 1000]`, `i64`, `float32` `beta`; `NT = 16`. -/
def setup_varlen_h4_d64_n3 : Inst := canon .i64 false 1 1000 4 4 64 64 3 16 true

theorem setup_varlen_h4_d64_n3_ok : setup_varlen_h4_d64_n3.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:761`, in `float16` there, here in `bfloat16`: `B = 1`, `T = 2000`, `H = 4`, `HV = 4`, `K = 100`, `V = 100`, packed, `cu_seqlens = [0, 15, 100, 300, 1200, 2000]`, `i64`, `float32` `beta`; `NT = 35`. -/
def setup_varlen_h4_d100_n5 : Inst := canon .i64 false 1 2000 4 4 100 100 5 35 true

theorem setup_varlen_h4_d100_n5_ok : setup_varlen_h4_d100_n5.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:762`, in `float16` there, here in `bfloat16`: `B = 1`, `T = 1000`, `H = 2`, `HV = 4`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 256, 500, 1000]`, `i64`, `float32` `beta`; `NT = 16`. -/
def setup_varlen_h2_hv4_d64_n3 : Inst := canon .i64 false 1 1000 2 4 64 64 3 16 true

theorem setup_varlen_h2_hv4_d64_n3_ok : setup_varlen_h2_hv4_d64_n3.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn.py:763`, in `float16` there, here in `bfloat16`: `B = 1`, `T = 1000`, `H = 2`, `HV = 8`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 256, 500, 1000]`, `i64`, `float32` `beta`; `NT = 16`. -/
def setup_varlen_h2_hv8_d64_n3 : Inst := canon .i64 false 1 1000 2 8 64 64 3 16 true

theorem setup_varlen_h2_hv8_d64_n3_ok : setup_varlen_h2_hv8_d64_n3.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:1099, 1102` (`test_recompute_w_u_fwd_varlen`; the test's `bfloat16` `beta`, line 1120): `B = 1`, `T = 128`, `H = 2`, `HV = 2`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 64, 128]`, `i64`, `bfloat16` `beta`; `NT = 2`. -/
def setup_wu_varlen_h2_d64 : Inst := canon .i64 true 1 128 2 2 64 64 2 2 true

theorem setup_wu_varlen_h2_d64_ok : setup_wu_varlen_h2_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:1100`: `B = 1`, `T = 256`, `H = 2`, `HV = 4`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 128, 256]`, `i64`, `bfloat16` `beta`; `NT = 4`. -/
def setup_wu_varlen_h2_hv4_d64 : Inst := canon .i64 true 1 256 2 4 64 64 2 4 true

theorem setup_wu_varlen_h2_hv4_d64_ok : setup_wu_varlen_h2_hv4_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `tests/ops/test_gdn_kernels.py:1101`: `B = 1`, `T = 128`, `H = 1`, `HV = 1`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 64, 128]`, `i64`, `bfloat16` `beta`; `NT = 2`. -/
def setup_wu_varlen_h1_d64 : Inst := canon .i64 true 1 128 1 1 64 64 2 2 true

theorem setup_wu_varlen_h1_d64_ok : setup_wu_varlen_h1_d64.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- the packed example of `chunk_gated_delta_rule`'s docstring (`fla/ops/gated_delta_rule/chunk.py:491, 502`): `B = 1`, `T = 8192`, `H = 4`, `HV = 8`, `K = 512`, `V = 512`, packed, `cu_seqlens = [0, 2048, 4096, 6144, 8192]`, `i64`, `bfloat16` `beta`; `NT = 128`. -/
def setup_doc_varlen_h4_hv8_d512 : Inst := canon .i64 true 1 8192 4 8 512 512 4 128 true

theorem setup_doc_varlen_h4_hv8_d512_ok : setup_doc_varlen_h4_hv8_d512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- Qwen3-Next's heads over a packed batch with `int32` `cu_seqlens`: `B = 1`, `T = 16384`, `H = 16`, `HV = 32`, `K = 128`, `V = 128`, packed, `cu_seqlens = [0, 1, 777, 2048, 4097, 8192, 8256, 12000, 16384]`, `i32`, `float32` `beta`; `NT = 260`. -/
def setup_qwen3next_varlen_i32 : Inst := canon .i32 false 1 16384 16 32 128 128 8 260 true

theorem setup_qwen3next_varlen_i32_ok : setup_qwen3next_varlen_i32.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- one token, `K = V = 1`: `B = 1`, `T = 1`, `H = 1`, `HV = 1`, `K = 1`, `V = 1`, padded, `float32` `beta`; `NT = 1`. -/
def setup_edge_t1_k1_v1 : Inst := canon .i64 false 1 1 1 1 1 1 1 1 false

theorem setup_edge_t1_k1_v1_ok : setup_edge_t1_k1_v1.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- packed, with empty sequences: `B = 1`, `T = 70`, `H = 2`, `HV = 2`, `K = 64`, `V = 64`, packed, `cu_seqlens = [0, 0, 5, 5, 70]`, `i32`, `float32` `beta`; `NT = 3`. -/
def setup_edge_varlen_empty : Inst := canon .i32 false 1 70 2 2 64 64 4 3 true

theorem setup_edge_varlen_empty_ok : setup_edge_varlen_empty.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- packed, 132 one-token sequences: `B = 1`, `T = 132`, `H = 2`, `HV = 4`, `K = 128`, `V = 64`, packed, `cu_seqlens = [0, 1, …, 132]`, `i64`, `float32` `beta`; `NT = 132`. -/
def setup_edge_varlen_132x1 : Inst := canon .i64 false 1 132 2 4 128 64 132 132 true

theorem setup_edge_varlen_132x1_ok : setup_edge_varlen_132x1.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- fla's `GatedDeltaProduct` defaults (`fla/layers/gated_deltaproduct.py:38-51`: 6 heads, `K = 256`, `V = 512`, 2 Householder steps, so `T = 2·2048` interleaved tokens; `bfloat16` `beta`, line 229, through `fla/ops/gated_delta_product/chunk.py:82`): `B = 2`, `T = 4096`, `H = 6`, `HV = 6`, `K = 256`, `V = 512`, padded, `bfloat16` `beta`; `NT = 64`. -/
def setup_gdp_b2_t4096_h6_k256_v512 : Inst := canon .i64 true 2 4096 6 6 256 512 2 64 false

theorem setup_gdp_b2_t4096_h6_k256_v512_ok : setup_gdp_b2_t4096_h6_k256_v512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- `GatedDeltaProduct` over a packed batch (`cu_seqlens · 2`, `fla/ops/gated_delta_product/chunk.py:42`): `B = 1`, `T = 4096`, `H = 6`, `HV = 6`, `K = 256`, `V = 512`, packed, `cu_seqlens = [0, 1000, 2600, 4096]`, `i64`, `bfloat16` `beta`; `NT = 65`. -/
def setup_gdp_varlen_h6_k256_v512 : Inst := canon .i64 true 1 4096 6 6 256 512 3 65 true

theorem setup_gdp_varlen_h6_k256_v512_ok : setup_gdp_varlen_h6_k256_v512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- fla's `PrecondGatedDeltaNet` defaults (`fla/layers/precond_gated_deltanet.py:106-109`; `bfloat16` `beta`, line 321, through `fla/ops/precond_gated_delta_rule/chunk.py:87`): `B = 2`, `T = 2048`, `H = 6`, `HV = 6`, `K = 256`, `V = 512`, padded, `bfloat16` `beta`; `NT = 32`. -/
def setup_precond_b2_t2048_h6_k256_v512 : Inst := canon .i64 true 2 2048 6 6 256 512 2 32 false

theorem setup_precond_b2_t2048_h6_k256_v512_ok : setup_precond_b2_t2048_h6_k256_v512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-- fla's `MoM` defaults (`fla/layers/mom.py:298-308`: 4 heads, `K = 256`, `V = 512`), the memories' tokens packed (`bfloat16` `beta`, lines 498, 503, 596, 736): `B = 1`, `T = 4096`, `H = 4`, `HV = 4`, `K = 256`, `V = 512`, packed, `cu_seqlens = [0, 700, 1500, 2048, 2900, 4096]`, `i64`, `bfloat16` `beta`; `NT = 66`. -/
def setup_mom_varlen_h4_k256_v512 : Inst := canon .i64 true 1 4096 4 4 256 512 5 66 true

theorem setup_mom_varlen_h4_k256_v512_ok : setup_mom_varlen_h4_k256_v512.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)

/-! ## Satisfiable -/

/-- Some covered instance and input exist: the one-token launch with every byte `0`. -/
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hk : setup_edge_t1_k1_v1.k.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hv : setup_edge_t1_k1_v1.v.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hb : setup_edge_t1_k1_v1.beta.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hA : setup_edge_t1_k1_v1.A.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hg : setup_edge_t1_k1_v1.g.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hw : setup_edge_t1_k1_v1.w.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hu : setup_edge_t1_k1_v1.u.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_edge_t1_k1_v1, canon])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
      (by norm_num [setup_edge_t1_k1_v1, canon, DType.bytes])
  have hok := setup_edge_t1_k1_v1_ok
  refine ⟨setup_edge_t1_k1_v1, zeroMem, hok,
    ⟨hk, hv, fun _ => hb, fun h => absurd h (by decide), hA, hg, hw, hu, ?_, ?_, ?_,
      fun h => absurd h (by decide), ?_, ?_, fun h => absurd h (by decide)⟩⟩
  · intro f h c hf hh hc
    have : kIdx setup_edge_t1_k1_v1 f h c < setup_edge_t1_k1_v1.k.len := by
      simp only [setup_edge_t1_k1_v1, canon, kIdx, nTok] at hf hh hc ⊢; omega
    exact ⟨_, zeroMem_bfAt hk this, zero_kvok⟩
  · intro f h c hf hh hc
    have : vIdx setup_edge_t1_k1_v1 f h c < setup_edge_t1_k1_v1.v.len := by
      simp only [setup_edge_t1_k1_v1, canon, vIdx, nTok] at hf hh hc ⊢; omega
    exact ⟨_, zeroMem_bfAt hv this, zero_kvok⟩
  · intro _ f h hf hh
    have : hIdx setup_edge_t1_k1_v1 f h < setup_edge_t1_k1_v1.beta.len := by
      simp only [setup_edge_t1_k1_v1, canon, hIdx, nTok, Bool.false_eq_true, ↓reduceIte] at hf hh ⊢
      omega
    exact ⟨_, zeroMem_f32At hb this, zero_betaok⟩
  · intro f h hf hh
    have : hIdx setup_edge_t1_k1_v1 f h < setup_edge_t1_k1_v1.g.len := by
      simp only [setup_edge_t1_k1_v1, canon, hIdx, nTok] at hf hh ⊢; omega
    exact ⟨_, zeroMem_f32At hg this, zero_gok⟩
  · intro f h j hf hh hj
    have : aIdx setup_edge_t1_k1_v1 f h j < setup_edge_t1_k1_v1.A.len := by
      simp only [setup_edge_t1_k1_v1, canon, aIdx, nTok] at hf hh hj ⊢; omega
    exact ⟨_, zeroMem_bfAt hA this, zero_aok⟩

end Claims.RecomputeWU
