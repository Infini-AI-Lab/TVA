import TVA

/-!
# `chunk_fwd_o_bf16`: chunked linear-attention output, fla

The kernel is `chunk_fwd_kernel_o` of `source/chunk_o.py` (lines 83–181), which calls the
`@triton.jit` helpers `unflatten_program_id` (lines 69–76) and `exp2` (lines 79–80); it reads the
module-level names `triton` and `tl` (lines 65–66). The host `chunk_fwd_o` (lines 223–273)
launches it on `bfloat16` `q`, `k` (`[B, T, H, K]`), `v` (`[B, T, HV, V]`), the states `h`
(`[B, NT, HV, K, V]`, or `[B, NT, HV, V, K]` with `state_v_first`), the `float32` gates `g`
(`[B, T, HV]`, or none) or the per-head decays `g_gamma` (`[HV]`, `float32`, or none), and writes
`o = torch.empty_like(v)`, on the grid `(cdiv(V, BV) · NT, B · HV)` with `BT = 64` and one of the
configurations `BK = BV = 32`, `64` or `128` of the source's list (`SETUP.md`); over fixed-length
rows (`cu_seqlens = None`) or a packed variable-length batch (`B = 1`, `cu_seqlens` and
`chunk_indices` of one dtype, `int32` or `int64`).

What is claimed: every covered launch ends normally; for every chunk `r` of the batch (a 64-token
chunk of a row, or the chunk a row of `chunk_indices` names), every token `t` in it, value head `hv`
and column `e`, the output element `o[t, hv, e]` holds a `bfloat16` value which, when its inputs
are finite and the magnitudes leave room (`InRange`), is finite and within `tol` of

  `scale · ( 2^g[t] · Σ_k q[t, hq, k] · S_r[hv, k, e]
             + Σ_{s in r's chunk, s ≤ t} (Σ_k q[t, hq, k] · k[s, hq, k]) · 2^(g[t] − g[s]) · v[s, hv, e] )`

(`hq = hv // (HV // H)`, `S_r` the chunk's state in `h`; with `g_gamma` instead of `g`, the gates
`2^(γ[hv]·(i+1))` and `2^(γ[hv]·(i−i'))` of the local positions `i`, `i'` of `t`, `s` in the
chunk; the gates `1` without either); every other byte is unchanged.

The conditions: `BK = BV ∈ {32, 64, 128}`; at least one row and one head, `H ∣ HV`, `K ≤ 256`,
`V ≤ 512`; `T` an `int32` (and `T + 63` does not wrap on a fixed-length batch of two rows or more),
at most `2^22` tokens, at most `8192` rows or sequences; at most one kind of gate; `B = 1` on
variable length; the grid within CUDA's limits; the buffers hold the tensors and `o` shares no byte
with the inputs; `int32` or `int64` tables, built as `prepare_chunk_indices` builds them
(`TablesOk`); a finite scale in `[2^-64, 2^64]`.

The tolerance rests on the device facts for `tl.dot` of `bfloat16` operands and for `tl.exp2` on
`float32` (`Device`).
-/

namespace Claims.ChunkFwdO

open TVA

/-! ## Constants of the source and the configuration -/

/-- `BT = chunk_size = 64` (the host's default, every covered launch). -/
def BT : Nat := 64

/-- The `float32` `+0` (`other=0.0`, the `tl.where` value `0` beside a `float32` tensor). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `tl.dot` configuration of lines 154, 156, 158 and 180: `bfloat16` operands, the default
`float32` accumulator, the default `input_precision` (`tf32`; no effect on 16-bit operands) and
the default `max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `chunk_fwd_kernel_o` (source lines 83–181) with `unflatten_program_id` (lines 69–76) inlined at
line 108 (`X = tl.cdiv(V, BV)`, a compile-time value; `Y` its default `None`: the compile-time
`if Y is None` takes the `pid % X, pid // X` branch) and `exp2` (lines 79–80) inlined at lines
164–165 and 169–170 (`x.to(tl.float32)` of a `float32` tensor is no conversion; `tl.math.exp2` is `tl.exp2`).
The pointers `q`, `k`, `v`, `h`, `o` are `*bf16`, `g` and `g_gamma` are `*fp32` (`None` without
them: never read); `cu_seqlens` and `chunk_indices` are pointers of the table dtype `idxD` (`int32`
or `int64`; `None` on a fixed-length launch, never read); `scale`
is the runtime `float32` scalar (`FloatV.hostF32` of the Python `float`); `T` is the runtime
`int32` (`do_not_specialize=['T']`). `H`, `HV`, `K`, `V`, `BK`, `BV` and the flags are
`tl.constexpr`, Lean parameters (the host passes `BK = BV`, `32`, `64` or `128`); `BT` is the
constant above; the flags' branches are compile-time `if`s, written as Lean `if`s. -/
def chunk_fwd_kernel_o (q k v h : Ptr .bf16) (g g_gamma : Ptr .f32) (o : Ptr .bf16) (idxD : DType)
    (cu_seqlens chunk_indices : Ptr idxD) (scale : F32) (T : I32) (H HV K V : Nat) (BK BV : Nat)
    (USE_G USE_G_GAMMA STATE_V_FIRST IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 108 (lines 72 and 74): `pid = tl.program_id(0).to(tl.int64)`, then `pid % X, pid // X`
    -- with the compile-time `X = tl.cdiv(V, BV)` built in `int64`
    let pid := tl.program_id ctx 0
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar pid)).item
    let i_v := (← tl.irem (Tensor.scalar pid)
      (Tensor.scalar (IntV.ofNat ((V + (BV - 1)) / BV) : I64))).item
    let i_t := (← tl.idiv (Tensor.scalar pid)
      (Tensor.scalar (IntV.ofNat ((V + (BV - 1)) / BV) : I64))).item
    -- line 109: axis 1, `.to(tl.int64)`
    let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
    -- line 110: `int64` `//` and `%` by the constexpr `HV`
    let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    -- lines 112–121: the compile-time `if IS_VARLEN:`; each branch gives the state's chunk index
    -- `i_tg`, the chunk index `i_t` within the sequence, `bos`, and the `int64` value line 133
    -- compares with (`T`)
    let (i_tg, i_t, bos, T) ← if IS_VARLEN then do
        -- line 113
        let i_tg := i_t
        -- line 114: `chunk_indices[i_t, 0].to(tl.int32)`, `chunk_indices[i_t, 1].to(tl.int64)`,
        -- both with the old `i_t` (the right side is evaluated before the assignment)
        let i_n := (← tl.convert idxD .i32 (Tensor.scalar
          (← tl.load1 (chunk_indices + i_t * (IntV.ofNat 2 : I64))))).item
        let i_t := (← tl.convert idxD .i64 (Tensor.scalar
          (← tl.load1 (chunk_indices + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
        -- line 115: the sequence's bounds, `.to(tl.int64)`
        let bos := (← tl.convert idxD .i64 (Tensor.scalar (← tl.load1 (cu_seqlens + i_n)))).item
        let eos := (← tl.convert idxD .i64
          (Tensor.scalar (← tl.load1 (cu_seqlens + i_n + (IntV.ofNat 1 : I32))))).item
        -- line 116: `T = eos - bos`, the wrapping `int64` subtraction (line 117's `NT` is not used
        -- on this path)
        pure (i_tg, i_t, bos, eos - bos)
      else do
        -- line 119: `tl.cdiv(T, BT)`, the `int32` `(T + 63) // 64`
        let NT := (← tl.idiv (Tensor.scalar (T + (IntV.ofNat (BT - 1) : I32)))
          (Tensor.scalar (IntV.ofNat BT : I32))).item
        -- line 120: `i_b * NT + i_t`, `NT` promoted to `int64`
        let NT64 := (← tl.convert .i32 .i64 (Tensor.scalar NT)).item
        -- line 121: `bos = i_b * T`, the `int32` `T` promoted to `int64`; `eos` is not used on
        -- this path, and `T` stays the argument, which line 133 promotes to `int64` the same way
        let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
        pure (i_b * NT64 + i_t, i_t, i_b * T64, T64)
    -- lines 123–124: the `int64` offsets in elements; `i_h // (HV // H)` with the compile-time
    -- `HV // H` built in `int64`
    let q := q + (bos * (IntV.ofNat H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
        (IntV.ofNat K : I64)
    let k := k + (bos * (IntV.ofNat H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
        (IntV.ofNat K : I64)
    -- lines 125–126
    let v := v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
    let o := o + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
    -- line 127: `(i_tg * HV + i_h).to(tl.int64) * K*V`: the `int64` value (`.to(tl.int64)` of an
    -- `int64` is no conversion), times `K`, times `V`
    let h := h + (i_tg * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
    -- lines 129–130
    let b_o := tl.zeros [BT, BV] .f32
    let b_A := tl.zeros [BT, BT] .f32
    -- line 132: the `int64` `i_t * BT` plus `tl.arange(0, BT)` (`int32`, promoted to `int64`)
    let o_t : Tensor [BT] I64 :=
      bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    -- line 133: the signed `int64` comparison
    let m_t := o_t.ltS T
    -- line 134
    let o_v : Tensor [BV] I64 :=
      bc (i_v * (IntV.ofNat BV : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BV))
    -- line 135: `range(tl.cdiv(K, BK))`, a compile-time bound (built in `int32`); the loop
    -- carries `b_o`, `b_A`
    let (b_o, b_A) ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + (BK - 1)) / BK) : I32)
      (IntV.ofInt 1 : I32) (b_o, b_A) fun i_k (b_o, b_A) => do
        -- line 136: `int32`
        let o_k : Tensor [BK] I32 := i_k * (IntV.ofNat BK : I32) + tl.arange 0 BK
        -- line 137
        let m_k := o_k.ltS (IntV.ofNat K : I32)
        -- line 138: `q + o_t[:, None] * (H*K)` (`int64`), then `+ o_k[None, :]` (`int32`)
        let p_q : Tensor [BT, BK] (Ptr .bf16) :=
          q + o_t.col BK * (IntV.ofNat (H * K) : I64) + o_k.row BT
        -- line 139
        let p_k : Tensor [BK, BT] (Ptr .bf16) :=
          k + o_k.col BT + o_t.row BK * (IntV.ofNat (H * K) : I64)
        -- lines 147–149: `other=0.0`, the `float32` `+0` converted to `bfloat16`
        let b_q ← tl.load p_q (some ((m_t.col BK).and (m_k.row BT)))
          (some (← tl.convert .f32 .bf16 (tl.full [BT, BK] zf32)))
        let b_k ← tl.load p_k (some ((m_k.col BT).and (m_t.row BK)))
          (some (← tl.convert .f32 .bf16 (tl.full [BK, BT] zf32)))
        -- lines 140–145 (the pointers and mask of the state, computed before line 147: no memory
        -- access), 150 (its load) and 155–158 (the compile-time `if STATE_V_FIRST:`)
        let b_o ← if STATE_V_FIRST then do
            -- lines 141–142: `h + o_v[:, None] * K` (`int64`), then `+ o_k[None, :]`
            let p_h : Tensor [BV, BK] (Ptr .bf16) :=
              h + o_v.col BK * (IntV.ofNat K : I64) + o_k.row BV
            let m_h := ((o_v.col BK).ltS (IntV.ofNat V : I64)).and (m_k.row BV)
            -- line 150
            let b_h ← tl.load p_h (some m_h)
              (some (← tl.convert .f32 .bf16 (tl.full [BV, BK] zf32)))
            -- line 154: `tl.dot(b_q, tl.trans(b_h), b_o)`
            tl.dot cfgBF16 b_q b_h.trans b_o
          else do
            -- lines 144–145: `h + o_k[:, None] * V` (`int32`), then `+ o_v[None, :]`
            let p_h : Tensor [BK, BV] (Ptr .bf16) :=
              h + o_k.col BV * (IntV.ofNat V : I32) + o_v.row BK
            let m_h := (m_k.col BV).and ((o_v.row BK).ltS (IntV.ofNat V : I64))
            -- line 150
            let b_h ← tl.load p_h (some m_h)
              (some (← tl.convert .f32 .bf16 (tl.full [BK, BV] zf32)))
            -- line 156: `tl.dot(b_q, b_h, b_o)`
            tl.dot cfgBF16 b_q b_h b_o
        -- line 158: `tl.dot(b_q, b_k, b_A)`
        let b_A ← tl.dot cfgBF16 b_q b_k b_A
        pure (b_o, b_A)
    -- lines 160–165: the compile-time `if USE_G:`
    let (b_o, b_A) ← if USE_G then do
        -- line 161: the `int64` offset
        let g := g + (bos * (IntV.ofNat HV : I64) + i_h)
        -- line 162
        let p_g : Tensor [BT] (Ptr .f32) := g + o_t * (IntV.ofNat HV : I64)
        -- line 163: `other=0.0`, the `float32` `+0`
        let b_g ← tl.load p_g (some m_t) (some (bc zf32))
        -- line 164: `exp2(b_g)[:, None]`, the IEEE `float32` product
        let b_o ← tl.mul b_o ((← tl.exp2 b_g).col BV)
        -- line 165: the IEEE `-` (no product feeds it), `exp2`, the IEEE product
        let b_A ← tl.mul b_A (← tl.exp2 (← tl.sub (b_g.col BT) (b_g.row BT)))
        pure (b_o, b_A)
      else pure (b_o, b_A)
    -- lines 166–170: the compile-time `if USE_G_GAMMA:`
    let (b_o, b_A) ← if USE_G_GAMMA then do
        -- line 167: `g_gamma[i_h]`, one `float32` (the `int64` offset)
        let b_gamma ← tl.load1 (g_gamma + i_h)
        -- line 168: `tl.arange(0, BT) + 1` (`int32`, the Python `1` built in `int32`), converted to
        -- `float32` for the product with the `float32` scalar (exact: at most 64); the IEEE product
        let b_g : Tensor [BT] F32 ← tl.mul (bc b_gamma)
          (← tl.convert .i32 .f32 ((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)))
        -- line 169: `exp2(b_g)[:, None]`, the IEEE product
        let b_o ← tl.mul b_o ((← tl.exp2 b_g).col BV)
        -- line 170: `b_g[:, None] - b_g[None, :]`: products feed it, so it may fuse (`tl.subc`);
        -- `exp2`, the IEEE product
        let b_A ← tl.mul b_A (← tl.exp2 (← tl.subc (b_g.col BT) (b_g.row BT)))
        pure (b_o, b_A)
      else pure (b_o, b_A)
    -- line 171: `o_t[:, None] >= o_t[None, :]` (signed `int64`), `&` the row and column masks
    let m_A := (Tensor.le (o_t.row BT) (o_t.col BT)).and ((m_t.col BT).and (m_t.row BT))
    -- line 172: the Python `0` built in `float32`
    let b_A := tl.where m_A b_A (bc zf32)
    -- lines 174–175
    let p_v : Tensor [BT, BV] (Ptr .bf16) := v + o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
    let p_o : Tensor [BT, BV] (Ptr .bf16) := o + o_t.col BV * (IntV.ofNat (HV * V) : I64) + o_v.row BT
    -- lines 177, 181: the mask `m_t[:, None] & (o_v < V)[None, :]`
    let m_o := (m_t.col BV).and ((o_v.ltS (IntV.ofNat V : I64)).row BT)
    -- line 177: `other=0.0` converted to `bfloat16`
    let b_v ← tl.load p_v (some m_o) (some (← tl.convert .f32 .bf16 (tl.full [BT, BV] zf32)))
    -- line 180: `b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale`: the IEEE products, then
    -- the addition two products feed (`tl.addc`); `b_A.to(bfloat16)` rounds to nearest even; the
    -- dot has the default zero accumulator and its result is multiplied, not added (no combine)
    let b_o ← tl.addc (← tl.mul b_o (bc scale))
      (← tl.mul (← tl.dot cfgBF16 (← tl.cvt b_A) b_v (tl.zeros [BT, BV] .f32)) (bc scale))
    -- line 181: the store's conversion to the pointer's `bfloat16` (nearest even)
    let o16 : Tensor [BT, BV] BF16 ← tl.cvt b_o
    tl.store p_o o16 (some m_o)

/-! ## The conditions -/

/-- What the host chooses: one launch through `chunk_fwd_o(q, k, v, h, g, g_gamma, scale,
state_v_first, cu_seqlens, chunk_size=64, chunk_indices)`, with the contents of its input
tensors. -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `B, T, H, K = q.shape` (`k` the same), `HV, V = v.shape[2], v.shape[3]`. -/
  B : Nat
  T : Nat
  H : Nat
  HV : Nat
  K : Nat
  V : Nat
  /-- `USE_G = g is not None`. -/
  useG : Bool
  /-- `USE_G_GAMMA = g_gamma is not None`. -/
  useGamma : Bool
  /-- `STATE_V_FIRST = state_v_first`. -/
  svf : Bool
  /-- The configuration: `BK = BV = bk` (the host's argument `BK`; `32`, `64` or `128`, with 2, 4
  or 8 warps). -/
  bk : Nat
  /-- `scale` (the Python `float`, a double; `K ** -0.5` by default, line 243). -/
  scale : F64
  /-- `q`, `k`: contiguous `[B, T, H, K]`; element `(b, t, i, x)` is `((b·T + t)·H + i)·K + x`. -/
  q : Buf .bf16
  k : Buf .bf16
  /-- `v`: contiguous `[B, T, HV, V]`; element `(b, t, j, e)` is `((b·T + t)·HV + j)·V + e`. -/
  v : Buf .bf16
  /-- `h`: contiguous `[B, NT, HV, K, V]` (`[B, NT, HV, V, K]` when `svf`), chunk `r` of the
  batch at `r·HV·K·V`. -/
  h : Buf .bf16
  /-- `g`: contiguous `[B, T, HV]` `float32` (unused when `useG` is false). -/
  g : Buf .f32
  /-- `g_gamma`: `[HV]` (or longer) `float32`, value head `j`'s decay at element `j` (unused when
  `useGamma` is false). -/
  gg : Buf .f32
  /-- `o = torch.empty_like(v)`. -/
  o : Buf .bf16
  /-- The contents of `q`, `k`, `v`, `h`, `g`. -/
  qv : Nat → BF16
  kv : Nat → BF16
  vv : Nat → BF16
  hv : Nat → BF16
  gv : Nat → F32
  ggv : Nat → F32
  /-- Variable length: the tables' dtype (`cu_seqlens`'s; `chunk_indices` carries it). -/
  idxD : DType
  /-- Variable length: the number of sequences `N` (`cu_seqlens` has `N + 1` entries). -/
  N : Nat
  /-- Variable length: `cu_seqlens` and its entries. -/
  cu : Buf idxD
  cuVal : Nat → idxD.Val
  /-- Variable length: the number of rows `NT = len(chunk_indices)`. -/
  NT : Nat
  /-- Variable length: `chunk_indices` (`[NT, 2]`, contiguous) and its entries. -/
  ci : Buf idxD
  ciVal : Nat → idxD.Val

/-- An integer table entry (`int32`/`int64`) as an integer. -/
def entryInt : (d : DType) → d.Val → Int
  | .i32, x => x.toInt
  | .i64, x => x.toInt
  | _, _ => 0

/-- `cu_seqlens[i]`. -/
def cuAt (d : Inst) (i : Nat) : Int := entryInt d.idxD (d.cuVal i)

/-- `chunk_indices[r, 0]` (the sequence) and `chunk_indices[r, 1]` (the chunk within it). -/
def seqAt (d : Inst) (r : Nat) : Int := entryInt d.idxD (d.ciVal (2 * r))
def chkAt (d : Inst) (r : Nat) : Int := entryInt d.idxD (d.ciVal (2 * r + 1))

/-- The chunks of a fixed-length row, `triton.cdiv(T, 64)` (line 241). -/
def NTpad (d : Inst) : Nat := (d.T + BT - 1) / BT

/-- The chunks of the batch, the states `h` holds: `B · cdiv(T, 64)` (fixed length) or
`len(chunk_indices)` (variable length). -/
def nRows (d : Inst) : Nat := if d.varlen then d.NT else d.B * NTpad d

/-- The grid's `x`: `cdiv(V, BV) · NT` (line 246), `NT = len(chunk_indices)` or `cdiv(T, 64)`. -/
def grid0 (d : Inst) : Nat :=
  (d.V + d.bk - 1) / d.bk * (if d.varlen then d.NT else NTpad d)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one row and one query/key head; `HV` value heads, a positive multiple of `H`
  (grouped value heads: `HV // H` value heads per query/key head). -/
  B_pos : 0 < d.B
  H_pos : 0 < d.H
  HV_pos : 0 < d.HV
  hv_dvd : d.H ∣ d.HV
  /-- The configuration: `BK = BV ∈ {32, 64, 128}`, the source's list on Hopper (SETUP.md). -/
  bk_ok : d.bk = 32 ∨ d.bk = 64 ∨ d.bk = 128
  /-- The head dims (SETUP.md: `K` up to 256, `V` up to 512). -/
  K_pos : 0 < d.K
  K_le : d.K ≤ 256
  V_pos : 0 < d.V
  V_le : d.V ≤ 512
  /-- `T` is a runtime `int32` (`do_not_specialize`: below `2^31`). -/
  T_lt : d.T < 2 ^ 31
  /-- On a fixed-length batch of two rows or more, `T + 63` does not wrap in line 119's `int32`
  `tl.cdiv` (**a limit of the kernel**, README, Exclusions; with one row `i_b = 0` makes `NT`
  unused, and a variable-length launch does not run line 119). -/
  T_wrap : d.varlen = false → 2 ≤ d.B → d.T + (BT - 1) < 2 ^ 31
  /-- `T` (a row's length, or the packed batch's total number of tokens) is a context length:
  at most `2^22` tokens (real model shapes, README, Exclusions). -/
  T_le : d.T ≤ 2 ^ 22
  /-- `B`, the batch rows, is a batch size: at most `8192` (real model shapes, README,
  Exclusions). -/
  B_le : d.B ≤ 8192
  /-- At most one kind of gate (fla's simple GLA: "Only one of `g` or `g_gamma` should be
  provided"). -/
  one_gate : ¬ (d.useG = true ∧ d.useGamma = true)
  /-- Variable length: `B = 1` (the callers' rule: the gated delta rule asserts it). -/
  varlen_B : d.varlen = true → d.B = 1
  /-- The grid within CUDA's limits (`x < 2^31`, `y = B·HV < 2^16`: **a limit of the kernel** on
  fixed-length batches, README, Exclusions) and fewer than `2^31` programs (`Launch.Valid`). -/
  grid0_lt : grid0 d < 2 ^ 31
  grid1_lt : d.B * d.HV < 2 ^ 16
  size_lt : grid0 d * (d.B * d.HV) < 2 ^ 31
  /-- The buffers hold the tensors. -/
  q_len : d.B * d.T * d.H * d.K ≤ d.q.len
  k_len : d.B * d.T * d.H * d.K ≤ d.k.len
  v_len : d.B * d.T * d.HV * d.V ≤ d.v.len
  o_len : d.B * d.T * d.HV * d.V ≤ d.o.len
  h_len : nRows d * d.HV * d.K * d.V ≤ d.h.len
  g_len : d.useG = true → d.B * d.T * d.HV ≤ d.g.len
  gg_len : d.useGamma = true → d.HV ≤ d.gg.len
  /-- `o` is a fresh allocation: it shares no byte with the inputs. -/
  o_q : Disjoint d.o.bytes d.q.bytes
  o_k : Disjoint d.o.bytes d.k.bytes
  o_v : Disjoint d.o.bytes d.v.bytes
  o_h : Disjoint d.o.bytes d.h.bytes
  o_g : d.useG = true → Disjoint d.o.bytes d.g.bytes
  o_gg : d.useGamma = true → Disjoint d.o.bytes d.gg.bytes
  /-- Variable length: `int32` or `int64` tables, `N < 2^31` (a sequence index is an `int32`
  after line 114's `.to(tl.int32)`), the tables' lengths, and `o` sharing no byte with them. -/
  idx_ok : d.varlen = true → d.idxD = .i32 ∨ d.idxD = .i64
  N_lt : d.varlen = true → d.N < 2 ^ 31
  /-- Variable length: `N`, the number of sequences, is a batch size: at most `8192` (real model
  shapes, README, Exclusions). -/
  N_le : d.varlen = true → d.N ≤ 8192
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  ci_len : d.varlen = true → 2 * d.NT ≤ d.ci.len
  o_cu : d.varlen = true → Disjoint d.o.bytes d.cu.bytes
  o_ci : d.varlen = true → Disjoint d.o.bytes d.ci.bytes
  /-- `scale` is a finite double in `[2^-64, 2^64]` (by default `K ** -0.5 ∈ [1/16, 1]`). -/
  scale_ok : d.scale.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ d.scale.toRat ∧ d.scale.toRat ≤ 2 ^ 64

/-- The tables of a variable-length launch, as `prepare_chunk_indices` builds them (or the caller
passes the same table in any row order, with `h`'s chunks in that order): `cu_seqlens`
nonnegative, nondecreasing and at most `T`; each row of `chunk_indices` names a sequence `n < N`
and a chunk `c ≥ 0` that starts inside it (`64·c < cu_seqlens[n+1] − cu_seqlens[n]`); no two rows
are the same; and every such chunk has a row. -/
def TablesOk (d : Inst) : Prop :=
  0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧ cuAt d d.N ≤ d.T ∧
  (∀ r, r < d.NT → 0 ≤ seqAt d r ∧ seqAt d r < d.N ∧ 0 ≤ chkAt d r ∧
    chkAt d r * BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) ∧
  (∀ r r', r < d.NT → r' < d.NT → r ≠ r' → (seqAt d r, chkAt d r) ≠ (seqAt d r', chkAt d r')) ∧
  (∀ n c : Nat, n < d.N → (c * BT : Int) < cuAt d (n + 1) - cuAt d n →
    ∃ r, r < d.NT ∧ seqAt d r = n ∧ chkAt d r = c)

/-- The inputs covered: the buffers are valid and hold the instance's contents (`g`, `g_gamma`
when the launch has them); on a variable-length launch, the tables are `TablesOk`. The values are covered element by
element in `Post` (`InRange`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.v.Valid init ∧ d.h.Valid init ∧ d.o.Valid init ∧
    d.q.Holds init d.qv ∧ d.k.Holds init d.kv ∧ d.v.Holds init d.vv ∧ d.h.Holds init d.hv ∧
    (d.useG = true → d.g.Valid init ∧ d.g.Holds init d.gv) ∧
    (d.useGamma = true → d.gg.Valid init ∧ d.gg.Holds init d.ggv) ∧
    (d.varlen = true → d.cu.Valid init ∧ d.ci.Valid init ∧ d.cu.Holds init d.cuVal ∧
      d.ci.Holds init d.ciVal ∧ TablesOk d)

/-- The model on an instance (`BT = 64`, `BK = BV = bk`). -/
def model (d : Inst) : Kernel :=
  chunk_fwd_kernel_o (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.h.ptr 0) (d.g.ptr 0) (d.gg.ptr 0)
    (d.o.ptr 0) d.idxD (d.cu.ptr 0) (d.ci.ptr 0) (FloatV.hostF32 d.scale) (IntV.ofNat d.T)
    d.H d.HV d.K d.V d.bk d.bk d.useG d.useGamma d.svf d.varlen

/-- The host's launch (lines 245–273): the grid `(cdiv(V, BV) · NT, B · HV)`; the source's block
shapes (`[BT]`: `o_t`, `m_t`, `b_g`; `[BK] = [BV]`: `o_v`, `o_k`, `m_k`; `[BT, BT]`: `b_A`, `m_A`;
`[BT, BK]`, `[BK, BT]`: `b_q`, `b_k`, `b_o`, `b_v`, their pointers and masks; `[BK, BV]`: the state;
the expanded `[BT, 1]`, `[1, BT]`, `[BK, 1]`, `[1, BK]`); `enable_fp_fusion` Triton's default
(`true`), `debug = false`, `num_ctas = 1`, no `launch_pdl`. `num_warps` (2, 4 or 8) and
`num_stages = 3` change no meaning of a model without a conflict inside a program. -/
def launch (d : Inst) : Launch :=
  { grid := ![grid0 d, d.B * d.HV, 1], fusion := true,
    blocks := [[BT], [d.bk], [BT, BT], [BT, d.bk], [d.bk, BT], [d.bk, d.bk], [BT, 1], [1, BT],
      [d.bk, 1], [1, d.bk]] }

/-! ## The specification -/

/-- Chunk `r`'s tokens are `[rowLo r, rowHi r)` of the flat token axis (`b·T + t` for row `b`):
for a fixed-length batch, `r = b·NT + c` is chunk `c` of row `b`; for a variable-length batch,
row `r` of `chunk_indices` names chunk `chkAt r` of sequence `seqAt r`. -/
def rowLo (d : Inst) (r : Nat) : Nat :=
  if d.varlen then (cuAt d (seqAt d r).toNat).toNat + BT * (chkAt d r).toNat
  else r / NTpad d * d.T + BT * (r % NTpad d)
def rowHi (d : Inst) (r : Nat) : Nat :=
  if d.varlen then min (rowLo d r + BT) (cuAt d ((seqAt d r).toNat + 1)).toNat
  else min (rowLo d r + BT) (r / NTpad d * d.T + d.T)

/-- The query/key head of value head `j`: `j // (HV // H)` (grouped value heads). -/
def qh (d : Inst) (j : Nat) : Nat := j / (d.HV / d.H)

/-- `q[t, i, x]`, `k[t, i, x]`, `v[t, j, e]`, `g[t, j]` at flat token `t`, as rationals. -/
def qAt (d : Inst) (t i x : Nat) : ℚ := (d.qv ((t * d.H + i) * d.K + x)).toRat
def kAt (d : Inst) (t i x : Nat) : ℚ := (d.kv ((t * d.H + i) * d.K + x)).toRat
def vAt (d : Inst) (t j e : Nat) : ℚ := (d.vv ((t * d.HV + j) * d.V + e)).toRat
def gAt (d : Inst) (t j : Nat) : ℚ := (d.gv (t * d.HV + j)).toRat

/-- The index in `h` of chunk `r`'s state `S_r[j, x, e]` (value head `j`, key dim `x`, value dim
`e`): `h[r, j, x, e]`, or `h[r, j, e, x]` when `svf`. -/
def hIdx (d : Inst) (r j x e : Nat) : Nat :=
  if d.svf then ((r * d.HV + j) * d.V + e) * d.K + x else ((r * d.HV + j) * d.K + x) * d.V + e
def hAt (d : Inst) (r j x e : Nat) : ℚ := (d.hv (hIdx d r j x e)).toRat

/-- The index of `o[t, j, e]`. -/
def oIdx (d : Inst) (t j e : Nat) : Nat := (t * d.HV + j) * d.V + e

/-- `liftMag` of a `bfloat16` value (`Device.liftMag`: a subnormal counted at `2^-126`). -/
def L (x : ℚ) : ℚ := Device.liftMag .bf16 x

/-- The inter-chunk dot `Σ_x q[t, qh j, x] · S_r[j, x, e]` and its lifted magnitude. -/
def QS (d : Inst) (r t j e : Nat) : ℚ :=
  ∑ x ∈ Finset.range d.K, qAt d t (qh d j) x * hAt d r j x e
def PQS (d : Inst) (r t j e : Nat) : ℚ :=
  ∑ x ∈ Finset.range d.K, L (qAt d t (qh d j) x) * L (hAt d r j x e)

/-- The score `Σ_x q[t, qh j, x] · k[s, qh j, x]` and its lifted magnitude. -/
def Aqk (d : Inst) (t s j : Nat) : ℚ :=
  ∑ x ∈ Finset.range d.K, qAt d t (qh d j) x * kAt d s (qh d j) x
def PA (d : Inst) (t s j : Nat) : ℚ :=
  ∑ x ∈ Finset.range d.K, L (qAt d t (qh d j) x) * L (kAt d s (qh d j) x)

/-- `g_gamma[j]`, value head `j`'s decay (base 2), as a rational. -/
def gam (d : Inst) (j : Nat) : ℚ := (d.ggv j).toRat

/-- The gates of token `t` (and `s`) of chunk `r`, value head `j`: `2^g[t, j]` and
`2^(g[t, j] − g[s, j])` with `g`; with `g_gamma`, `2^(γ_j·(i+1))` and `2^(γ_j·(i − i'))`, `i = t −
rowLo r` and `i' = s − rowLo r` the local positions (`b_gamma * (tl.arange(0, BT) + 1)`); `1`
without either. -/
noncomputable def gate (d : Inst) (r t j : Nat) : ℝ :=
  if d.useG then (2 : ℝ) ^ ((gAt d t j : ℚ) : ℝ)
  else if d.useGamma then (2 : ℝ) ^ ((gam d j * ((t - rowLo d r : Nat) + 1 : ℚ) : ℚ) : ℝ) else 1
noncomputable def wgt (d : Inst) (_r t s j : Nat) : ℝ :=
  if d.useG then (2 : ℝ) ^ ((gAt d t j - gAt d s j : ℚ) : ℝ)
  else if d.useGamma then (2 : ℝ) ^ ((gam d j * ((t : ℚ) - s) : ℚ) : ℝ) else 1

/-- `scale` as a real. -/
noncomputable def sig (d : Inst) : ℝ := (d.scale.toRat : ℝ)

/-- The tokens of `t`'s chunk up to `t`. -/
def upTo (d : Inst) (r t : Nat) : Finset Nat := Finset.Icc (rowLo d r) t

/-- **The exact output** at token `t` of chunk `r`, value head `j`, column `e`:
`scale · (2^g[t]·Σ_x q·S_r + Σ_{s ≤ t} (q[t]·k[s])·2^(g[t]−g[s])·v[s])`. -/
noncomputable def ref (d : Inst) (r t j e : Nat) : ℝ :=
  sig d * (gate d r t j * (QS d r t j e : ℝ) +
    ∑ s ∈ upTo d r t, (Aqk d t s j : ℝ) * wgt d r t s j * (vAt d s j e : ℝ))

/-- The intra-chunk term's magnitude, `Σ_{s ≤ t} |q[t]·k[s]|·2^(g[t]−g[s])·|v[s]|`: the scale of
the `bfloat16` rounding of the scores (line 180). -/
noncomputable def Mabs (d : Inst) (r t j e : Nat) : ℝ :=
  ∑ s ∈ upTo d r t, |(Aqk d t s j : ℝ)| * wgt d r t s j * |(vAt d s j e : ℝ)|

/-- The lifted magnitude the dots' errors are relative to:
`2^g[t]·PQS + Σ_{s ≤ t} 2^(g[t]−g[s])·PA·liftMag v[s]`. -/
noncomputable def Mag (d : Inst) (r t j e : Nat) : ℝ :=
  gate d r t j * (PQS d r t j e : ℝ) +
    ∑ s ∈ upTo d r t, wgt d r t s j * (PA d t s j : ℝ) * (L (vAt d s j e) : ℝ)

/-- The scale of the absolute floors (the dots', `exp2`'s and the conversions' absolute terms):
`1 + 2^g[t] + PQS + Σ_{s ≤ t} (1 + 2^(g[t]−g[s]) + PA)·liftMag v[s]`. -/
noncomputable def Flo (d : Inst) (r t j e : Nat) : ℝ :=
  1 + gate d r t j + (PQS d r t j e : ℝ) +
    ∑ s ∈ upTo d r t, (1 + wgt d r t s j + (PA d t s j : ℝ)) * (L (vAt d s j e) : ℝ)

/-- The relative error the gates' computation adds with `g_gamma`: the `float32` products
`γ_j·(i+1)` and their (fusable) difference move each exponent by at most `2^-15·|γ_j|`
(README, "Tolerances"); `0` otherwise. -/
noncomputable def gamErr (d : Inst) (j : Nat) : ℝ :=
  if d.useGamma then (1 / 2 ^ 15) * |((gam d j : ℚ) : ℝ)| else 0

/-- **The tolerance** (README, "Tolerances"): the store's `bfloat16` rounding of the result
(`2^-8` of `|ref|`, applied to the error too), the `bfloat16` rounding of the scores (`2^-8` of
`Mabs`), the dots', `exp2`'s and the `float32` roundings' relative errors (`2^-13` of `Mag`, for
each configuration: `relDot16 BK` over `cdiv(K, BK)` steps, README), the
`g_gamma` gates' exponents (`gamErr` of `Mag`), and absolute floors. -/
noncomputable def tol (d : Inst) (r t j e : Nat) : ℝ :=
  (1 / 2 ^ 8) * |ref d r t j e| +
    (1 + 1 / 2 ^ 8) * sig d * ((1 / 2 ^ 8) * Mabs d r t j e +
      (1 / 2 ^ 13 + gamErr d j) * Mag d r t j e + (1 / 2 ^ 100) * Flo d r t j e) + 1 / 2 ^ 130

/-- The element is in range: the values it reads are finite (the query row and state column, the
keys up to `t`, the value column over the whole chunk, the gates up to `t`); the gates satisfy
`|g[s]| ≤ 2^64`, `g[t] ≤ 64` and `g[t] − g[s] ≤ 64` (`s ≤ t`; gated DeltaNet's gates are
cumulative sums of nonpositive log decays: `g[t] ≤ g[s] ≤ 0`), and the decay `γ_j` is finite with
`|γ_j| ≤ 64` and `γ_j ≤ 1` (RetNet's and lightning attention's decays are in `[−12, 0]`); and the
magnitudes leave room in
`float32` (`2^100`, `float32` overflows near `2^128`). -/
def InRange (d : Inst) (r t j e : Nat) : Prop :=
  (∀ x, x < d.K → (d.qv ((t * d.H + qh d j) * d.K + x)).isFinite = true ∧
      (d.hv (hIdx d r j x e)).isFinite = true) ∧
    (∀ s ∈ upTo d r t, ∀ x, x < d.K → (d.kv ((s * d.H + qh d j) * d.K + x)).isFinite = true) ∧
    (∀ s, rowLo d r ≤ s → s < rowHi d r → (d.vv ((s * d.HV + j) * d.V + e)).isFinite = true) ∧
    (d.useGamma = true → (d.ggv j).isFinite = true ∧ |gam d j| ≤ 64 ∧ gam d j ≤ 1) ∧
    (d.useG = true → gAt d t j ≤ 64 ∧ ∀ s ∈ upTo d r t,
      (d.gv (s * d.HV + j)).isFinite = true ∧ |gAt d s j| ≤ 2 ^ 64 ∧ gAt d t j - gAt d s j ≤ 64) ∧
    PQS d r t j e ≤ 2 ^ 100 ∧ gate d r t j * (PQS d r t j e : ℝ) ≤ 2 ^ 100 ∧
    (∀ s ∈ upTo d r t, PA d t s j ≤ 2 ^ 100 ∧ wgt d r t s j * (PA d t s j : ℝ) ≤ 2 ^ 100) ∧
    Mag d r t j e + Flo d r t j e ≤ 2 ^ 100 ∧ sig d * (Mag d r t j e + Flo d r t j e) ≤ 2 ^ 100

/-- The output elements the launch writes: every token of every chunk, every value head and
column. -/
def written (d : Inst) : Set Nat :=
  {x | ∃ r t j e, r < nRows d ∧ rowLo d r ≤ t ∧ t < rowHi d r ∧ j < d.HV ∧ e < d.V ∧
    x = oIdx d t j e}

/-- What the launch leaves: for every chunk `r`, token `t` in it, value head `j` and column `e`,
`o[t, j, e]` holds a `bfloat16` value which, when the element is `InRange`, is finite and within
`tol` of `ref`; every byte outside the written elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r t j e, r < nRows d → rowLo d r ≤ t → t < rowHi d r → j < d.HV → e < d.V →
    ∃ c : BF16, fin.read .bf16 (d.o.ptr (oIdx d t j e)).addr = some c ∧
      (InRange d r t j e → c.isFinite = true ∧ |c.toReal - ref d r t j e| ≤ tol d r t j e)) ∧
    ∀ z, z ∉ d.o.elems (written d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `chunk_fwd_kernel_o` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## The host's layouts

`mk vl useG useGm svf B T H HV K V sbits dt N NT bk` is the host's launch for those sizes: contiguous
tensors, `o = empty_like(v)`, the storages at `2^40`, `2·2^40`, …, `7·2^40` (`chunk_indices` at
`7·2^40 + 2^39`), zero contents; `sbits` the pattern of the `float` scale, `dt` the tables' dtype,
`N` sequences and `NT` rows on a variable-length launch, `bk` the configuration (`BK = BV`, by
default `64`). -/

/-- The chunks of the batch for those sizes (`nRows`). -/
def rowsOf (vl : Bool) (B T NT : Nat) : Nat := if vl then NT else B * ((T + BT - 1) / BT)

def mk (vl useG useGm svf : Bool) (B T H HV K V sbits : Nat) (dt : DType) (N NT : Nat)
    (bk : Nat := 64) : Inst where
  varlen := vl
  B := B
  T := T
  H := H
  HV := HV
  K := K
  V := V
  useG := useG
  useGamma := useGm
  svf := svf
  bk := bk
  scale := FloatV.ofPat sbits
  q := ⟨2 ^ 40, B * T * H * K⟩
  k := ⟨2 * 2 ^ 40, B * T * H * K⟩
  v := ⟨3 * 2 ^ 40, B * T * HV * V⟩
  h := ⟨4 * 2 ^ 40, rowsOf vl B T NT * HV * K * V⟩
  g := ⟨5 * 2 ^ 40, B * T * HV⟩
  gg := ⟨8 * 2 ^ 40, HV⟩
  o := ⟨6 * 2 ^ 40, B * T * HV * V⟩
  qv := fun _ => FloatV.ofPat 0
  kv := fun _ => FloatV.ofPat 0
  vv := fun _ => FloatV.ofPat 0
  hv := fun _ => FloatV.ofPat 0
  gv := fun _ => FloatV.ofPat 0
  ggv := fun _ => FloatV.ofPat 0
  idxD := dt
  N := N
  cu := ⟨7 * 2 ^ 40, N + 1⟩
  cuVal := fun _ => dt.ofBits 0
  NT := NT
  ci := ⟨7 * 2 ^ 40 + 2 ^ 39, 2 * NT⟩
  ciVal := fun _ => dt.ofBits 0

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The scale's conditions. -/
def ScaleOk (sbits : Nat) : Prop :=
  (FloatV.ofPat sbits : F64).isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ (FloatV.ofPat sbits : F64).toRat ∧
    (FloatV.ofPat sbits : F64).toRat ≤ 2 ^ 64

/-- **Every host launch at these sizes is covered**: `B ≥ 1` rows, `H ≥ 1` query/key heads and a
multiple `HV` of them, `K ≤ 256`, `V ≤ 512`, `T + 63 < 2^31`, `T ≤ 2^22`, `B ≤ 8192`, the grid within its
limits, each tensor at most `2^40` bytes, `int32` or `int64` tables of at most `8192` sequences, a scale in
`[2^-64, 2^64]`, and each configuration `BK = BV ∈ {32, 64, 128}`. -/
theorem mk_ok {vl useG useGm svf : Bool} {B T H HV K V sbits : Nat} {dt : DType} {N NT bk : Nat}
    (hB : 0 < B) (hH : 0 < H) (hHV : 0 < HV) (hdvd : H ∣ HV) (hK0 : 0 < K) (hK : K ≤ 256)
    (hV0 : 0 < V) (hV : V ≤ 512) (hT : T + (BT - 1) < 2 ^ 31) (hvB : vl = true → B = 1)
    (hone : ¬ (useG = true ∧ useGm = true))
    (hg0 : grid0 (mk vl useG useGm svf B T H HV K V sbits dt N NT bk) < 2 ^ 31) (hg1 : B * HV < 2 ^ 16)
    (hsz : grid0 (mk vl useG useGm svf B T H HV K V sbits dt N NT bk) * (B * HV) < 2 ^ 31)
    (hq : B * T * H * K * 2 ≤ 2 ^ 40) (hv : B * T * HV * V * 2 ≤ 2 ^ 40)
    (hgb : B * T * HV * 4 ≤ 2 ^ 40)
    (hh : rowsOf vl B T NT * HV * K * V * 2 ≤ 2 ^ 40) (hdt : dt = .i32 ∨ dt = .i64)
    (hN : N < 2 ^ 31) (hNsz : (N + 1) * 8 ≤ 2 ^ 39) (hNT : 2 * NT * 8 ≤ 2 ^ 39)
    (hsc : ScaleOk sbits) (hbk : bk = 32 ∨ bk = 64 ∨ bk = 128 := by decide)
    (hT22 : T ≤ 2 ^ 22 := by norm_num) (hB13 : B ≤ 8192 := by norm_num)
    (hN13 : N ≤ 8192 := by norm_num) :
    (mk vl useG useGm svf B T H HV K V sbits dt N NT bk).ok := by
  have hbytes : dt.bytes ≤ 8 := by rcases hdt with h | h <;> subst h <;> decide
  have hcu : ((N + 1 : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : (N + 1) * dt.bytes ≤ (N + 1) * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this hNsz
  have hci : ((2 * NT : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : 2 * NT * dt.bytes ≤ 2 * NT * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this hNT
  have hq' : ((B * T * H * K : Nat) : Int) * 2 ≤ 2 ^ 40 := by exact_mod_cast hq
  have hv' : ((B * T * HV * V : Nat) : Int) * 2 ≤ 2 ^ 40 := by exact_mod_cast hv
  have hh' : ((rowsOf vl B T NT * HV * K * V : Nat) : Int) * 2 ≤ 2 ^ 40 := by exact_mod_cast hh
  have hg' : ((B * T * HV : Nat) : Int) * 4 ≤ 2 ^ 40 := by exact_mod_cast hgb
  have hb2 : DType.bf16.bytes = 2 := rfl
  have hb4 : DType.f32.bytes = 4 := rfl
  refine
    { B_pos := hB, H_pos := hH, HV_pos := hHV, hv_dvd := hdvd, bk_ok := hbk, K_pos := hK0, K_le := hK,
      V_pos := hV0, V_le := hV, T_lt := by simp only [mk]; omega, T_wrap := fun _ _ => hT, T_le := hT22, B_le := hB13, one_gate := hone,
      varlen_B := hvB, grid0_lt := hg0, grid1_lt := hg1,
      size_lt := hsz, q_len := le_rfl, k_len := le_rfl, v_len := le_rfl, o_len := le_rfl,
      h_len := ?_, g_len := fun _ => le_rfl, gg_len := fun _ => le_rfl,
      o_q := bytes_disj _ _ (Or.inr (by simp only [mk, hb2]; push_cast at hq' ⊢; omega)),
      o_k := bytes_disj _ _ (Or.inr (by simp only [mk, hb2]; push_cast at hq' ⊢; omega)),
      o_v := bytes_disj _ _ (Or.inr (by simp only [mk, hb2]; push_cast at hv' ⊢; omega)),
      o_h := bytes_disj _ _ (Or.inr (by simp only [mk, hb2]; push_cast at hh' ⊢; omega)),
      o_gg := fun _ => bytes_disj _ _ (Or.inl (by simp only [mk, hb2]; push_cast at hv' ⊢; omega)),
      o_g := fun _ => bytes_disj _ _ (Or.inr (by simp only [mk, hb4]; push_cast at hg' ⊢; omega)),
      idx_ok := fun _ => hdt, N_lt := fun _ => hN, N_le := fun _ => hN13, cu_len := fun _ => le_rfl,
      ci_len := fun _ => le_rfl,
      o_cu := fun _ => bytes_disj _ _ (Or.inl (by simp only [mk, hb2]; push_cast at hv' hcu ⊢; omega)),
      o_ci := fun _ => bytes_disj _ _ (Or.inl (by simp only [mk, hb2]; push_cast at hv' hci ⊢; omega)),
      scale_ok := hsc }
  simp only [nRows, NTpad, mk, rowsOf]
  rfl

/-! ## Satisfiable

One row of one token, one head, `K = V = 1`, no gates, the state not transposed, the scale `1/8`,
every input `0` (every byte of memory `0`). -/

/-- `1/8 = 64 ** -0.5`, the host's default scale at `K = 64` (the double `0x3fc0000000000000`). -/
def sc64 : Nat := 0x3fc0000000000000
/-- `128 ** -0.5` (the double `0x3fb6a09e667f3bcc`). -/
def sc128 : Nat := 0x3fb6a09e667f3bcc
/-- `256 ** -0.5 = 1/16` (the double `0x3fb0000000000000`). -/
def sc256 : Nat := 0x3fb0000000000000

/-- `60 ** -0.5` (the double `0x3fc08654a2d4f6db`). -/
def sc60 : Nat := 0x3fc08654a2d4f6db

theorem sc60_ok : ScaleOk sc60 := ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sc64_ok : ScaleOk sc64 := ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sc128_ok : ScaleOk sc128 := ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sc256_ok : ScaleOk sc256 := ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

def satInst : Inst := mk false false false false 1 1 1 1 1 1 sc64 .i32 0 0

theorem satInst_ok : satInst.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc64_ok

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem zero_valid {dt : DType} (b : Buf dt) (h0 : 0 ≤ b.base) (h1 : b.base + b.len * dt.bytes ≤ 2 ^ 64)
    (h2 : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun x _ => by simp [zeroMem]⟩

theorem zero_holds_bf16 (b : Buf .bf16) (h0 : 0 ≤ b.base) (h1 : b.base + b.len * 2 ≤ 2 ^ 64)
    (h2 : b.base % 2 = 0) : b.Holds zeroMem (fun _ => FloatV.ofPat 0) := by
  intro i hi
  have hal : Mem.Aligned .bf16 (b.ptr i).addr := by
    have hi' : (i : Int) + 1 ≤ b.len := by exact_mod_cast hi
    refine ⟨?_, ?_, ?_⟩ <;> simp only [Buf.ptr_addr, show DType.bf16.bytes = 2 from rfl] <;> omega
  unfold Mem.read
  simp only [hal, ↓reduceIte]
  rfl

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨satInst, zeroMem, satInst_ok, ?_⟩
  have hb2 : DType.bf16.bytes = 2 := rfl
  refine ⟨zero_valid _ (by simp [satInst, mk]) (by simp [satInst, mk, hb2]) (by simp [satInst, mk, hb2]),
    zero_valid _ (by simp [satInst, mk]) (by simp [satInst, mk, hb2]) (by simp [satInst, mk, hb2]),
    zero_valid _ (by simp [satInst, mk]) (by simp [satInst, mk, hb2]) (by simp [satInst, mk, hb2]),
    zero_valid _ (by simp [satInst, mk]) (by simp [satInst, mk, rowsOf, hb2, BT])
      (by simp [satInst, mk, hb2]),
    zero_valid _ (by simp [satInst, mk]) (by simp [satInst, mk, hb2]) (by simp [satInst, mk, hb2]),
    zero_holds_bf16 _ (by simp [satInst, mk]) (by simp [satInst, mk]) (by simp [satInst, mk]),
    zero_holds_bf16 _ (by simp [satInst, mk]) (by simp [satInst, mk]) (by simp [satInst, mk]),
    zero_holds_bf16 _ (by simp [satInst, mk]) (by simp [satInst, mk]) (by simp [satInst, mk]),
    zero_holds_bf16 _ (by simp [satInst, mk]) (by simp [satInst, mk, rowsOf, BT]) (by simp [satInst, mk]),
    fun h => by simp [satInst, mk] at h, fun h => by simp [satInst, mk] at h,
    fun h => by simp [satInst, mk] at h⟩

/-! ## Coverage of the setup's launches (SETUP.md, "The launches")

Names: `setup_<origin>`. The contents (and so `TablesOk` and `InRange`) are not part of
`Inst.ok`; `check_gpu.py` checks them on the GPU. -/

/-- fla's `test_chunk_fwd_o` (`tests/ops/test_gdn_kernels.py:451-479`), its `bfloat16` cases:
`(B, T, H, HV, D) = (2, 128, 2, 2, 64)` with gates, `(2, 128, 2, 4, 64)` with gates,
`(2, 128, 2, 2, 64)` without; fixed length, `state_v_first = False`. -/
def setup_test_g : Inst := mk false true false false 2 128 2 2 64 64 sc64 .i32 0 0
theorem setup_test_g_ok : setup_test_g.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc64_ok
def setup_test_gva : Inst := mk false true false false 2 128 2 4 64 64 sc64 .i32 0 0
theorem setup_test_gva_ok : setup_test_gva.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc64_ok
def setup_test_nog : Inst := mk false false false false 2 128 2 2 64 64 sc64 .i32 0 0
theorem setup_test_nog_ok : setup_test_nog.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc64_ok

/-- fla's `test_chunk_o_deterministic` (`tests/ops/test_gdn_kernels.py:706-722`): one packed batch
of `8192` tokens, `cu_seqlens = [0, 5000, 8192]` (`int32`), 8 heads, `K = V = 128`, gates;
`79 + 50 = 129` chunks. -/
def setup_det_varlen : Inst := mk true true false false 1 8192 8 8 128 128 sc128 .i32 2 129
theorem setup_det_varlen_ok : setup_det_varlen.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- Qwen3-Next's gated DeltaNet layers through fla's layer (`fla/layers/gated_deltanet.py:302-317`:
`state_v_first=True`, `int32` `cu_seqlens`): 16 query/key heads, 32 value heads, `K = V = 128`,
gates; a packed prefill of 8 prompts of 4096 tokens (512 chunks). -/
def setup_qwen3next : Inst := mk true true false true 1 32768 16 32 128 128 sc128 .i32 8 512
theorem setup_qwen3next_ok : setup_qwen3next.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- The same heads at the size of a long-context training batch: `2^20` tokens in 256 sequences
of 4096 (16384 chunks). -/
def setup_qwen3next_1m : Inst := mk true true false true 1 1048576 16 32 128 128 sc128 .i32 256 16384
theorem setup_qwen3next_1m_ok : setup_qwen3next_1m.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- fla's gated DeltaNet defaults (`fla/models/gated_deltanet/configuration_gated_deltanet.py:26-33`:
`head_dim = 256`, `expand_v = 2`, 6 heads: `K = 256`, `V = 512`), `state_v_first=True`, a packed
batch with `int64` `cu_seqlens` (fla's varlen tests build them with `torch.LongTensor`):
`[0, 15, 100, 300, 1200, 2000]`, `1 + 2 + 4 + 15 + 13 = 35` chunks. -/
def setup_gdn_default : Inst := mk true true false true 1 2000 6 6 256 512 sc256 .i64 5 35
theorem setup_gdn_default_ok : setup_gdn_default.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
    (by norm_num) sc256_ok

/-- fla's op benchmark shapes (`benchmarks/ops/registry.py:203-209`) on fixed-length batches:
`B1_T8192_H96_D128` with gates, `B4_T4096_H64_D128` without (the delta rule, `g=None`,
`fla/ops/delta_rule/chunk.py:51-61`), `B8_T2048_H32_D256` with gates. -/
def setup_bench_h96 : Inst := mk false true false false 1 8192 96 96 128 128 sc128 .i32 0 0
theorem setup_bench_h96_ok : setup_bench_h96.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_bench_h64_nog : Inst := mk false false false false 4 4096 64 64 128 128 sc128 .i32 0 0
theorem setup_bench_h64_nog_ok : setup_bench_h64_nog.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_bench_d256 : Inst := mk false true false false 8 2048 32 32 256 256 sc256 .i32 0 0
theorem setup_bench_d256_ok : setup_bench_d256.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc256_ok

/-- fla's head dims that are not multiples of 64 (`tests/ops/test_gdn.py:75,159,161`: 60 and
100), fixed length, gates, `K = 60`, `V = 100`. -/
def setup_dims_60_100 : Inst := mk false true false false 2 500 4 4 60 100 sc60 .i32 0 0
theorem setup_dims_60_100_ok : setup_dims_60_100.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc60_ok

/-- The largest fixed-length batch below the grid's `y` limit at Qwen3-Next's 32 value heads:
`B = 2047` (`B·HV = 65504`), one token per row, `state_v_first`. -/
def setup_grid_edge : Inst := mk false true false true 2047 1 16 32 128 128 sc128 .i32 0 0
theorem setup_grid_edge_ok : setup_grid_edge.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- RetNet through fla's `chunk_retention` (`fla/ops/retention/chunk.py:68-79`: `g_gamma =
log(1 − 2^(−5−h))`, which `chunk_simple_gla` scales by `1/ln 2`, `fla/ops/simple_gla/chunk.py:180-181`)
at fla's multiscale retention defaults (`fla/layers/multiscale_retention.py:74-77`: hidden size
1024, `expand_k = 1`, `expand_v = 2`, 8 heads: `K = 128`, `V = 256`), a fixed-length batch of
2 × 4096 tokens, and a packed batch `[0, 1000, 3000, 8000, 8192]` with `int64` tables (130 chunks). -/
def setup_retnet : Inst := mk false false true false 2 4096 8 8 128 256 sc128 .i32 0 0
theorem setup_retnet_ok : setup_retnet.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_retnet_varlen : Inst := mk true false true false 1 8192 8 8 128 256 sc128 .i64 4 130
theorem setup_retnet_varlen_ok : setup_retnet_varlen.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- Lightning attention through fla's `chunk_lightning_attn` (`fla/ops/lightning_attn/chunk.py:73-85`:
`g_gamma = −(8/H)·(1 − layer_idx/num_layers)·h`, scaled by `1/ln 2`) at fla's benchmark shapes
(`benchmarks/ops/registry.py:203-209, 576-582`, `layer_idx = 0`, `num_layers = 12`):
`B1_T8192_H96_D128` and `B4_T2048_H16_D128`. -/
def setup_lightning_h96 : Inst := mk false false true false 1 8192 96 96 128 128 sc128 .i32 0 0
theorem setup_lightning_h96_ok : setup_lightning_h96.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_lightning_h16 : Inst := mk false false true false 4 2048 16 16 128 128 sc128 .i32 0 0
theorem setup_lightning_h16_ok : setup_lightning_h16.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

/-- The other two configurations of the source's list, `BK = BV = 32` (2 warps) and `BK = BV = 128`
(8 warps), which the autotuner picks on some shapes (`SETUP.md`: `BK = BV = 128` on fla's benchmark
shapes with 64 and 96 heads): the launches above at those configurations. -/
def setup_test_g_bk32 : Inst := mk false true false false 2 128 2 2 64 64 sc64 .i32 0 0 32
theorem setup_test_g_bk32_ok : setup_test_g_bk32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc64_ok
def setup_det_varlen_bk32 : Inst := mk true true false false 1 8192 8 8 128 128 sc128 .i32 2 129 32
theorem setup_det_varlen_bk32_ok : setup_det_varlen_bk32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_dims_60_100_bk32 : Inst := mk false true false false 2 500 4 4 60 100 sc60 .i32 0 0 32
theorem setup_dims_60_100_bk32_ok : setup_dims_60_100_bk32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc60_ok
def setup_bench_h96_bk128 : Inst := mk false true false false 1 8192 96 96 128 128 sc128 .i32 0 0 128
theorem setup_bench_h96_bk128_ok : setup_bench_h96_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_bench_h64_nog_bk128 : Inst := mk false false false false 4 4096 64 64 128 128 sc128 .i32 0 0 128
theorem setup_bench_h64_nog_bk128_ok : setup_bench_h64_nog_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_bench_d256_bk128 : Inst := mk false true false false 8 2048 32 32 256 256 sc256 .i32 0 0 128
theorem setup_bench_d256_bk128_ok : setup_bench_d256_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc256_ok
def setup_qwen3next_bk128 : Inst := mk true true false true 1 32768 16 32 128 128 sc128 .i32 8 512 128
theorem setup_qwen3next_bk128_ok : setup_qwen3next_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_qwen3next_1m_bk128 : Inst := mk true true false true 1 1048576 16 32 128 128 sc128 .i32 256 16384 128
theorem setup_qwen3next_1m_bk128_ok : setup_qwen3next_1m_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_gdn_default_bk128 : Inst := mk true true false true 1 2000 6 6 256 512 sc256 .i64 5 35 128
theorem setup_gdn_default_bk128_ok : setup_gdn_default_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
    (by norm_num) sc256_ok
def setup_dims_60_100_bk128 : Inst := mk false true false false 2 500 4 4 60 100 sc60 .i32 0 0 128
theorem setup_dims_60_100_bk128_ok : setup_dims_60_100_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc60_ok
def setup_retnet_bk128 : Inst := mk false false true false 2 4096 8 8 128 256 sc128 .i32 0 0 128
theorem setup_retnet_bk128_ok : setup_retnet_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok
def setup_lightning_h96_bk128 : Inst := mk false false true false 1 8192 96 96 128 128 sc128 .i32 0 0 128
theorem setup_lightning_h96_bk128_ok : setup_lightning_h96_bk128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num [BT]) (by simp) (by decide) (by decide) (by norm_num)
    (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
    (by norm_num) sc128_ok

end Claims.ChunkFwdO
