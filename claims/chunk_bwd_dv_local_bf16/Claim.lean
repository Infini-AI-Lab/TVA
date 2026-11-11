import TVA

/-!
# `chunk_bwd_dv_local_bf16`: intra-chunk value gradient of chunked linear attention, fla

The kernel is `chunk_bwd_kernel_dv_local` of `source/chunk_o.py` (lines 74–149), which calls the
`@triton.jit` helper `exp2` (lines 70–71); it reads the module-level names `triton` and `tl`
(lines 66–67). The host `chunk_bwd_dv_local` (lines 191–241) launches it on `bfloat16` `q`, `k`
(`[B, T, H, K]`) and `do` (`[B, T, HV, V]`), an optional `float32` gate `g` (`[B, T, HV]`), and
writes `dv = torch.empty_like(do)`, over fixed-length rows (`cu_seqlens is None`) or a packed
variable-length batch (`B = 1`, `cu_seqlens` and `chunk_indices` of one dtype, `int32` or
`int64`); one program per (chunk, batch row and value head), on the grid `(NT, B·HV)`.

What is claimed: every covered launch ends normally; for every sequence (a row of a fixed-length
batch, or `[cu_seqlens[n], cu_seqlens[n+1])`), every token `p` of it, every value head `hv` and
every column `e < V`, `dv[p, hv, e]` is a finite `bfloat16` within `tol` of the exact

  `Σ_{j = p}^{end(p) − 1} σ · (Σ_c k[p, h, c]·q[j, h, c]) · 2^{g[j, hv] − g[p, hv]} · do[j, hv, e]`

(`end(p)` the end of `p`'s chunk of `BT` tokens, counted from the sequence's start,
`h = hv // (HV // H)`, `σ` the host's double `scale`, the gate factor `1` without a gate); every
other byte is unchanged.

The conditions: `chunk_size` is `16`, `32` or `64` and `BK`, `BV` the host's powers of two in
`[16, 128]`; `1 ≤ K ≤ 256`, `1 ≤ V ≤ 2^16`, `HV·V < 2^31`; `H ∣ HV`; `T` an `int32` and at most
`2^22` tokens, at most `8192` rows or sequences, `B = 1` on variable length; the grid within
CUDA's limits; a finite scale in `[2^-32, 2^32]`; the buffers hold the tensors and `dv` shares no
byte with the inputs; `int32` or `int64` tables, built as `prepare_chunk_indices` builds them
(`TablesOk`); and the input values finite with room in `float32` (`ValsOk`).

The tolerance rests on the device facts for `tl.dot` of `bfloat16` operands and for `tl.exp2` on
`float32` (`Device`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-! ## Constants of the source -/

/-- The `float32` `+0` (`other=0.0`; the `tl.where` value `0` beside a `float32` block). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `tl.dot` configuration of lines 136 and 148: `bfloat16` operands, the default `float32`
accumulator, the default `input_precision` (`tf32`; no effect on 16-bit operands) and the default
`max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `chunk_bwd_kernel_dv_local` (source lines 74–149) with `exp2` (lines 70–71) inlined at line
138. `q`, `k`, `do_` (the source's `do`), `dv` are `*bf16`, `g` is `*fp32` (`None`, never read,
when `USE_G` is false); `cu_seqlens`, `chunk_indices` are pointers of the tables' dtype `idxD`
(`None`, never read, on a fixed-length launch); `scale` is the runtime `float32` scalar (a Python
`float` argument: `FloatV.hostF32` of the double); `T` is the runtime `int32`
(`do_not_specialize=['T']`). `H`, `HV`, `K`, `V`, `BT`, `BK`, `BV` and the flags are
`tl.constexpr`, Lean parameters; `USE_G_GAMMA` and `USE_A` are the compile-time `False` on every
covered launch (`g_gamma = A = None`), so the model has no parameter for them and their branches
(lines 116–117, 124–125) are not visited. -/
def chunk_bwd_kernel_dv_local (q k : Ptr .bf16) (g : Ptr .f32) (do_ dv : Ptr .bf16)
    (idxD : DType) (cu_seqlens chunk_indices : Ptr idxD) (scale : F32) (T : I32)
    (H HV K V BT BK BV : Nat) (USE_G IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 99: the program ids, `.to(tl.int64)` (exact widenings)
    let i_t := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
    -- line 100: `int64` `//` and `%` by the constexpr `HV` (built in `int64`)
    let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    -- lines 101–106: the compile-time `if IS_VARLEN:`; each branch gives the chunk index `i_t`,
    -- `bos` and the length line 114 compares with (as an `int64`)
    let (i_t, bos, T) ← if IS_VARLEN then do
        -- line 102: both loads read the old `i_t`; `.to(tl.int32)`, `.to(tl.int64)`
        let i_n := (← tl.convert idxD .i32 (Tensor.scalar
          (← tl.load1 (chunk_indices + i_t * (IntV.ofNat 2 : I64))))).item
        let i_t := (← tl.convert idxD .i64 (Tensor.scalar
          (← tl.load1 (chunk_indices + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
        -- line 103
        let bos := (← tl.convert idxD .i64 (Tensor.scalar (← tl.load1 (cu_seqlens + i_n)))).item
        let eos := (← tl.convert idxD .i64
          (Tensor.scalar (← tl.load1 (cu_seqlens + i_n + (IntV.ofNat 1 : I32))))).item
        -- line 104: `T = eos - bos`, the wrapping `int64` subtraction
        pure (i_t, bos, eos - bos)
      else do
        -- line 106: the `int32` `T` promoted to `int64`; `eos` is not read on this path
        let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
        pure (i_t, i_b * T64, T64)
    -- lines 108–109: `i_h // (HV // H)` (`HV // H` Python's, at compile time, built in `int64`)
    let q := q + (bos * (IntV.ofNat H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
        (IntV.ofNat K : I64)
    let k := k + (bos * (IntV.ofNat H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
        (IntV.ofNat K : I64)
    -- lines 110–111
    let do_ := do_ + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
    let dv := dv + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
    -- line 113: the `int64` `i_t * BT` plus `tl.arange(0, BT)` (`int32`, promoted to `int64`)
    let o_t : Tensor [BT] I64 :=
      bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    -- line 114: the signed `int64` comparison with the scalar `T`
    let m_t := o_t.ltS T
    -- lines 119–122: the compile-time `if USE_G:`; the gate block (never read without a gate)
    let b_g : Tensor [BT] F32 ← if USE_G then do
        let g := g + (bos * (IntV.ofNat HV : I64) + i_h)
        let p_g := g + o_t * (IntV.ofNat HV : I64)
        tl.load p_g (some m_t) (some (bc zf32))
      else pure (bc zf32)
    -- line 127
    let b_A := tl.zeros [BT, BT] .f32
    -- line 128: `range(tl.cdiv(K, BK))`, the bound Python's, at compile time (built in `int32`);
    -- the loop carries `b_A`
    let b_A ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + BK - 1) / BK) : I32)
      (IntV.ofInt 1 : I32) b_A fun i_k b_A => do
        -- lines 129–130: `int32`
        let o_k : Tensor [BK] I32 := i_k * (IntV.ofNat BK : I32) + tl.arange 0 BK
        let m_k := o_k.ltS (IntV.ofNat K : I32)
        -- lines 131–132: the pointers advanced by the `int64` `o_t * (H*K)` and the `int32` `o_k`
        let p_k := k + (o_t * (IntV.ofNat (H * K) : I64)).col BK + o_k.row BT
        let p_q := q + o_k.col BT + (o_t * (IntV.ofNat (H * K) : I64)).row BK
        -- lines 134–135: `other=0.0` is the `float32` `+0`, converted to the pointer's dtype
        let b_k : Tensor [BT, BK] BF16 ← tl.load p_k (some ((m_t.col BK).and (m_k.row BT)))
          (some (← tl.convert .f32 .bf16 (tl.full [BT, BK] zf32)))
        let b_q : Tensor [BK, BT] BF16 ← tl.load p_q (some ((m_k.col BT).and (m_t.row BK)))
          (some (← tl.convert .f32 .bf16 (tl.full [BK, BT] zf32)))
        -- line 136: `tl.dot(b_k, b_q)` (the default zero accumulator) times the runtime scalar
        -- `scale`, added to `b_A`: a fusable multiply-add
        let b_kq : Tensor [BT, BT] F32 ← tl.dot cfgBF16 b_k b_q (tl.zeros [BT, BT] .f32)
        tl.mul_add b_kq (bc scale) b_A
    -- lines 137–138: `USE_G or USE_G_GAMMA` is `USE_G`; `b_g[None, :] - b_g[:, None]` (IEEE, no
    -- product feeds it), `exp2` (`.to(tl.float32)` of a `float32` block is no conversion,
    -- `tl.math.exp2`), the IEEE product
    let b_A ← if USE_G then do
        let b_dg ← tl.sub (b_g.row BT) (b_g.col BT)
        tl.mul b_A (← tl.exp2 b_dg)
      else pure b_A
    -- line 139
    let m_A := (Tensor.le (o_t.col BT) (o_t.row BT)).and ((m_t.col BT).and (m_t.row BT))
    -- line 140: the `float32` `+0`, then `.to(do.dtype.element_ty)`, one rounding to `bfloat16`
    let b_A : Tensor [BT, BT] BF16 ← tl.cvt (tl.where m_A b_A (bc zf32))
    -- line 142: `range(tl.cdiv(V, BV))`, no carried state
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((V + BV - 1) / BV) : I32)
      (IntV.ofInt 1 : I32) () fun i_v _ => do
        -- lines 143–146
        let o_v : Tensor [BV] I32 := i_v * (IntV.ofNat BV : I32) + tl.arange 0 BV
        let m_v := o_v.ltS (IntV.ofNat V : I32)
        let p_do := do_ + (o_t * (IntV.ofNat (HV * V) : I64)).col BV + o_v.row BT
        let p_dv := dv + (o_t * (IntV.ofNat (HV * V) : I64)).col BV + o_v.row BT
        -- line 147
        let b_do : Tensor [BT, BV] BF16 ← tl.load p_do (some ((m_t.col BV).and (m_v.row BT)))
          (some (← tl.convert .f32 .bf16 (tl.full [BT, BV] zf32)))
        -- line 148: `b_A.to(b_do.dtype)` of a `bfloat16` block is no conversion; the default zero
        -- accumulator
        let b_dv : Tensor [BT, BV] F32 ← tl.dot cfgBF16 b_A b_do (tl.zeros [BT, BV] .f32)
        -- line 149: the store's conversion to the pointer's `bfloat16` (nearest even)
        let o16 : Tensor [BT, BV] BF16 ← tl.cvt b_dv
        tl.store p_dv o16 (some ((m_t.col BV).and (m_v.row BT)))

/-! ## The conditions -/

/-- What the host chooses: one launch through `chunk_bwd_dv_local(q, k, do, g, g_gamma=None,
A=None, scale, cu_seqlens, chunk_size, chunk_indices)`, with the contents of the index tables.
The tensors are contiguous: element `(b, t, h, c)` of `q`, `k` is `((b·T + t)·H + h)·K + c`,
element `(b, t, hv, e)` of `do`, `dv` is `((b·T + t)·HV + hv)·V + e`, element `(b, t, hv)` of `g`
is `(b·T + t)·HV + hv`. -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `USE_G = g is not None`. -/
  useG : Bool
  /-- `B, T, H, K = k.shape` (`q` the same), `HV = do.shape[2]`, `V = do.shape[-1]`. -/
  B : Nat
  T : Nat
  H : Nat
  HV : Nat
  K : Nat
  V : Nat
  /-- `BT = chunk_size`. -/
  BT : Nat
  /-- `BK`, `BV`: the host's `min(max(next_power_of_2(K), 16), 128)` and the same for `V`. -/
  BK : Nat
  BV : Nat
  /-- `scale` (the Python `float`, a double). -/
  scale : F64
  /-- The storages of `q`, `k`, `do`, `dv` (`bfloat16`) and `g` (`float32`; unused without a
  gate). -/
  q : Buf .bf16
  k : Buf .bf16
  dO : Buf .bf16
  dv : Buf .bf16
  g : Buf .f32
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

/-- The grid's `x`: `NT = len(chunk_indices)` (variable length) or `triton.cdiv(T, BT)` (fixed
length). -/
def nChunks (d : Inst) : Nat := if d.varlen then d.NT else (d.T + d.BT - 1) / d.BT

/-- The grid's `y`: `B·HV`. -/
def nRows (d : Inst) : Nat := d.B * d.HV

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `chunk_size` is `16`, `32` or `64` (SETUP.md). -/
  bt_ok : d.BT = 16 ∨ d.BT = 32 ∨ d.BT = 64
  /-- `BK`, `BV` powers of two in `[16, 128]` (the host's values on an H200). -/
  bk_pow2 : ∃ a, d.BK = 2 ^ a
  bk_ge : 16 ≤ d.BK
  bk_le : d.BK ≤ 128
  bv_pow2 : ∃ a, d.BV = 2 ^ a
  bv_ge : 16 ≤ d.BV
  bv_le : d.BV ≤ 128
  /-- Head dims: `1 ≤ K ≤ 256` (SETUP.md: 32 to 256), `1 ≤ V`. -/
  K_pos : 0 < d.K
  K_le : d.K ≤ 256
  V_pos : 0 < d.V
  /-- `V ≤ 2^16` and `HV·V < 2^31` (SETUP.md: `V ≤ 1000`, `HV·V ≤ 12288`): the constexprs `V`
  (line 144) and `HV*V` (lines 145–146) are built as `int32` constants, as the model writes them
  (an `int` in `[2^31, 2^32)` would be a `uint32` constant, and line 144's comparison unsigned),
  and `o_v` (line 143) stays below `2^31`. -/
  V_le : d.V ≤ 2 ^ 16
  hvv_lt : d.HV * d.V < 2 ^ 31
  /-- Heads: `H ≥ 1` divides `HV ≥ 1` (grouped value heads). -/
  H_pos : 0 < d.H
  HV_pos : 0 < d.HV
  hv_dvd : d.H ∣ d.HV
  /-- At least one batch row; variable length: `B = 1` (fla's callers). -/
  B_pos : 0 < d.B
  varlen_B : d.varlen = true → d.B = 1
  /-- `T` is an `int32` argument (`do_not_specialize`: a runtime `int32` below `2^31`). -/
  T_lt : d.T < 2 ^ 31
  /-- `T` is a context length (the tokens of a row, or of the packed batch): at most `2^22`. -/
  T_ctx : d.T ≤ 2 ^ 22
  /-- `B`, the batch rows, is a batch size: at most `8192`. -/
  B_le : d.B ≤ 8192
  /-- The grid `(NT, B·HV)` within CUDA's limits: `NT < 2^31` and, **a limit of the kernel**, `B·HV`
  on the grid's `y` axis below `2^16` (README, Exclusions); fewer than `2^31` programs. -/
  grid0_lt : nChunks d < 2 ^ 31
  grid1_lt : nRows d < 2 ^ 16
  grid_lt : nChunks d * nRows d < 2 ^ 31
  /-- `scale` a finite double in `[2^-32, 2^32]` (by default `K^-0.5 ∈ [1/16, 1]`). -/
  scale : d.scale.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ d.scale.toRat ∧ d.scale.toRat ≤ 2 ^ 32
  /-- The buffers hold the tensors. -/
  q_len : d.B * d.T * d.H * d.K ≤ d.q.len
  k_len : d.B * d.T * d.H * d.K ≤ d.k.len
  dO_len : d.B * d.T * d.HV * d.V ≤ d.dO.len
  dv_len : d.B * d.T * d.HV * d.V ≤ d.dv.len
  g_len : d.useG = true → d.B * d.T * d.HV ≤ d.g.len
  /-- `dv` is a fresh allocation (`empty_like(do)`): it shares no byte with the inputs. -/
  dv_q : Disjoint d.dv.bytes d.q.bytes
  dv_k : Disjoint d.dv.bytes d.k.bytes
  dv_dO : Disjoint d.dv.bytes d.dO.bytes
  dv_g : d.useG = true → Disjoint d.dv.bytes d.g.bytes
  /-- Variable length: `int32` or `int64` tables, `N < 2^31` (a sequence index is an `int32` after
  line 102's `.to(tl.int32)`), the tables' lengths, and `dv` sharing no byte with them. -/
  idx_ok : d.varlen = true → d.idxD = .i32 ∨ d.idxD = .i64
  N_lt : d.varlen = true → d.N < 2 ^ 31
  /-- Variable length: `N`, the number of sequences, is a batch size: at most `8192`. -/
  N_le : d.varlen = true → d.N ≤ 8192
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  ci_len : d.varlen = true → 2 * d.NT ≤ d.ci.len
  dv_cu : d.varlen = true → Disjoint d.dv.bytes d.cu.bytes
  dv_ci : d.varlen = true → Disjoint d.dv.bytes d.ci.bytes

/-- The model on an instance. -/
def model (d : Inst) : Kernel :=
  chunk_bwd_kernel_dv_local (d.q.ptr 0) (d.k.ptr 0) (d.g.ptr 0) (d.dO.ptr 0) (d.dv.ptr 0) d.idxD
    (d.cu.ptr 0) (d.ci.ptr 0) (FloatV.hostF32 d.scale) (IntV.ofNat d.T) d.H d.HV d.K d.V d.BT
    d.BK d.BV d.useG d.varlen

/-- The host's launch (lines 213–240): the grid `(NT, B·HV)`; the source's block shapes (`[BT]`:
`o_t`, `m_t`, `b_g`; `[BK]`: `o_k`, `m_k`; `[BV]`: `o_v`, `m_v`; `[BT, BT]`: `b_A`, `m_A`;
`[BT, BK]`: `b_k`; `[BK, BT]`: `b_q`; `[BT, BV]`: `b_do`, `b_dv`; the expanded `[BT, 1]`,
`[1, BT]`, `[1, BK]`, `[BK, 1]`, `[1, BV]`); `enable_fp_fusion` Triton's default (`true`);
`num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps = 4` and `num_stages = 3` change no
outcome of a model without a conflict inside a program. -/
def launch (d : Inst) : Launch :=
  { grid := ![nChunks d, nRows d, 1], fusion := true,
    blocks := [[d.BT], [d.BK], [d.BV], [d.BT, d.BT], [d.BT, d.BK], [d.BK, d.BT], [d.BT, d.BV],
      [d.BT, 1], [1, d.BT], [1, d.BK], [d.BK, 1], [1, d.BV]] }

/-! ### Reading the inputs -/

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

/-- The `float32` value at element `i` of a buffer, if present. -/
def rdF (m : Mem) (b : Buf .f32) (i : Nat) : Option F32 := m.read .f32 (b.ptr (i : Int)).addr

/-- Element `i` of a `float32` buffer as a rational (`0` when absent). -/
def valF (m : Mem) (b : Buf .f32) (i : Nat) : ℚ :=
  match rdF m b i with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `float32` buffer is a finite value. -/
def finF (m : Mem) (b : Buf .f32) (i : Nat) : Prop :=
  ∃ x : F32, rdF m b i = some x ∧ x.isFinite = true

/-- Element `(tok, h, c)` of `q` or `k` (`tok` the flat token, `b·T + t`). -/
def qkIdx (d : Inst) (tok h c : Nat) : Nat := (tok * d.H + h) * d.K + c
/-- Element `(tok, hv, e)` of `do` or `dv`. -/
def oIdx (d : Inst) (tok hv e : Nat) : Nat := (tok * d.HV + hv) * d.V + e
/-- Element `(tok, hv)` of `g`. -/
def gIdx (d : Inst) (tok hv : Nat) : Nat := tok * d.HV + hv

/-- The query/key head of value head `hv`: `hv // (HV // H)`. -/
def hk (d : Inst) (hv : Nat) : Nat := hv / (d.HV / d.H)

/-! ### Sequences and chunks -/

/-- The number of sequences: the rows of a fixed-length batch, or `N`. -/
def nSeq (d : Inst) : Nat := if d.varlen then d.N else d.B

/-- Sequence `n`'s tokens are `[lo n, hi n)` of the flat token axis (`b·T + t` for row `b`). -/
def lo (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d n).toNat else n * d.T
def hi (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d (n + 1)).toNat else n * d.T + d.T

/-- The start of token `p`'s chunk, chunks of `BT` tokens counted from the sequence's start, and
its end, `min(start + BT, hi n)`. -/
def chunkStart (d : Inst) (n p : Nat) : Nat := lo d n + (p - lo d n) / d.BT * d.BT
def chunkEnd (d : Inst) (n p : Nat) : Nat := min (chunkStart d n p + d.BT) (hi d n)

/-- The tables of a variable-length launch, as `prepare_chunk_indices` builds them (or the caller
passes the same table): `cu_seqlens` nonnegative, nondecreasing and at most `T`; each row of
`chunk_indices` names a sequence `n < N` and a chunk `c ≥ 0` that starts inside it
(`c·BT < cu_seqlens[n+1] − cu_seqlens[n]`); no two rows are the same; and every such chunk has a
row. -/
def TablesOk (d : Inst) : Prop :=
  0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧ cuAt d d.N ≤ d.T ∧
  (∀ r, r < d.NT → 0 ≤ seqAt d r ∧ seqAt d r < d.N ∧ 0 ≤ chkAt d r ∧
    chkAt d r * d.BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) ∧
  (∀ r r', r < d.NT → r' < d.NT → r ≠ r' → (seqAt d r, chkAt d r) ≠ (seqAt d r', chkAt d r')) ∧
  (∀ n c : Nat, n < d.N → (c * d.BT : Int) < cuAt d (n + 1) - cuAt d n →
    ∃ r, r < d.NT ∧ seqAt d r = n ∧ chkAt d r = c)

/-! ### The exact quantities -/

/-- `scale` as a real (the host's double). -/
noncomputable def sig (d : Inst) : ℝ := (d.scale.toRat : ℝ)

/-- `Σ_c k[p, h, c]·q[j, h, c]` for value head `hv` (`h = hk hv`), exact. -/
noncomputable def kq (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.K, ((valB m d.k (qkIdx d p (hk d hv) c) : ℚ) : ℝ) *
    ((valB m d.q (qkIdx d j (hk d hv) c) : ℚ) : ℝ)

/-- Its magnitude `Σ_c liftMag k · liftMag q` (`Device.liftMag .bf16`: a subnormal counted at
`2^-126`), the magnitude `Device.DotBF16`'s error is relative to; at least `|kq|`. -/
noncomputable def kqMag (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.K,
    ((Device.liftMag .bf16 (valB m d.k (qkIdx d p (hk d hv) c)) *
      Device.liftMag .bf16 (valB m d.q (qkIdx d j (hk d hv) c)) : ℚ) : ℝ)

/-- The gate difference `g[j, hv] − g[p, hv]`, exact. -/
noncomputable def dg (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  ((valF m d.g (gIdx d j hv) : ℚ) : ℝ) - ((valF m d.g (gIdx d p hv) : ℚ) : ℝ)

/-- The exact gate factor `2^{g[j] − g[p]}` (`1` without a gate). -/
noncomputable def wt (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  if d.useG then (2 : ℝ) ^ dg d m p j hv else 1

/-- The exact entry `A[p, j] = σ·kq·gate` of the masked matrix (`p ≤ j` in one chunk). -/
noncomputable def Aex (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  sig d * kq d m p j hv * wt d m p j hv

/-- `do[j, hv, e]`, and its lifted magnitude. -/
noncomputable def dov (d : Inst) (m : Mem) (j hv e : Nat) : ℝ :=
  ((valB m d.dO (oIdx d j hv e) : ℚ) : ℝ)
noncomputable def doMag (d : Inst) (m : Mem) (j hv e : Nat) : ℝ :=
  ((Device.liftMag .bf16 (valB m d.dO (oIdx d j hv e)) : ℚ) : ℝ)

/-- **The exact value gradient** at token `p` of sequence `n`, value head `hv`, column `e`:
`Σ_{j = p}^{chunkEnd − 1} A[p, j]·do[j, hv, e]`. -/
noncomputable def ref (d : Inst) (m : Mem) (n p hv e : Nat) : ℝ :=
  ∑ j ∈ Finset.Ico p (chunkEnd d n p), Aex d m p j hv * dov d m j hv e

/-! ### The tolerance (README, "Tolerances") -/

/-- `u = 2^-24`, one `float32` rounding. -/
noncomputable def u : ℝ := 1 / 2 ^ 24

/-- The gate's error: the computed `exp2(fl(g[j] − g[p]))` is within `wErr` of `2^{g[j] − g[p]}`
(the IEEE subtraction's rounding inside the exponent, `Device.Exp2F32`'s `2^-22` and `2^-126`); `0`
without a gate. `Pre`'s room bounds use it (`wUp`). -/
noncomputable def wErr (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  if d.useG then
    (2 : ℝ) ^ dg d m p j hv *
        ((2 : ℝ) ^ (u * |dg d m p j hv| + 1 / 2 ^ 149) * (1 + 1 / 2 ^ 22) - 1) + 1 / 2 ^ 126
  else 0

/-- The gate's upper bound, `wt + wErr`. -/
noncomputable def wUp (d : Inst) (m : Mem) (p j hv : Nat) : ℝ := wt d m p j hv + wErr d m p j hv

/-- The magnitude behind entry `(p, j)`: `σ·kqMag·wt`, the sum of the magnitudes of the products
`σ·k·q·2^{dg}` that make the entry, with floors for the absolute errors of the `tl.dot`s
(`kqMag + 2^-94`), of `exp2` (`wt + 2^-112`) and of the second `tl.dot` (`+ 2^-86`). -/
noncomputable def aMag (d : Inst) (m : Mem) (p j hv : Nat) : ℝ :=
  sig d * (kqMag d m p j hv + 1 / 2 ^ 94) * (wt d m p j hv + 1 / 2 ^ 112) + 1 / 2 ^ 86

/-- **The tolerance** at `(n, p, hv, e)`, over `j ∈ [p, chunkEnd)`:
`2^-7·Σ_j |Aex·do| + 2^-13·Σ_j aMag·liftMag do + 2^-100`. The first term is the kernel's two
`bfloat16` roundings (of each entry of `A`, line 140, and of the stored result, line 149),
relative to the magnitudes of `ref`'s terms; the second the `float32` errors of the entries
(the `K` loop's `tl.dot`s, the gate) and line 148's `tl.dot`, relative to the products'
magnitudes. -/
noncomputable def tol (d : Inst) (m : Mem) (n p hv e : Nat) : ℝ :=
  let J := Finset.Ico p (chunkEnd d n p)
  1 / 2 ^ 7 * ∑ j ∈ J, |Aex d m p j hv| * |dov d m j hv e| +
    1 / 2 ^ 13 * ∑ j ∈ J, aMag d m p j hv * doMag d m j hv e + 1 / 2 ^ 100

/-! ### The inputs covered -/

/-- The input values covered, in every sequence `n`:
* every component of `q`, `k`, `do` at its tokens finite, `|do| ≤ 2^100`, and with a gate every
  `g` finite;
* every pair `p ≤ j` of one chunk: with a gate `g[j] − g[p] ≤ 64` (`exp2` stays finite),
  `kqMag ≤ 2^100` and `σ·kqMag·(1 + wUp) ≤ 2^100` (room in `float32`);
* every output: `Σ_j σ·kqMag·wUp·liftMag do ≤ 2^120` (room for line 148's `tl.dot` and the
  store). -/
def ValsOk (d : Inst) (m : Mem) : Prop :=
  ∀ n, n < nSeq d →
    (∀ p, lo d n ≤ p → p < hi d n →
      (∀ h c, h < d.H → c < d.K → finB m d.q (qkIdx d p h c) ∧ finB m d.k (qkIdx d p h c)) ∧
      (∀ hv e, hv < d.HV → e < d.V →
        finB m d.dO (oIdx d p hv e) ∧ |valB m d.dO (oIdx d p hv e)| ≤ 2 ^ 100) ∧
      (d.useG = true → ∀ hv, hv < d.HV → finF m d.g (gIdx d p hv))) ∧
    (∀ p j hv, lo d n ≤ p → p ≤ j → j < chunkEnd d n p → hv < d.HV →
      (d.useG = true → dg d m p j hv ≤ 64) ∧ kqMag d m p j hv ≤ 2 ^ 100 ∧
      sig d * kqMag d m p j hv * (1 + wUp d m p j hv) ≤ 2 ^ 100) ∧
    (∀ p hv e, lo d n ≤ p → p < hi d n → hv < d.HV → e < d.V →
      ∑ j ∈ Finset.Ico p (chunkEnd d n p),
        sig d * kqMag d m p j hv * wUp d m p j hv * doMag d m j hv e ≤ 2 ^ 120)

/-- The inputs covered: the buffers are valid; on a variable-length launch the tables hold the
instance's contents and are `TablesOk`; the values are covered (`ValsOk`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.dO.Valid init ∧ d.dv.Valid init ∧
    (d.useG = true → d.g.Valid init) ∧
    (d.varlen = true → d.cu.Valid init ∧ d.ci.Valid init ∧ d.cu.Holds init d.cuVal ∧
      d.ci.Holds init d.ciVal ∧ TablesOk d) ∧
    ValsOk d init

/-! ## The specification -/

/-- The elements of `dv` the launch writes: every token of every sequence, every value head and
column. -/
def written (d : Inst) : Set Nat :=
  {x | ∃ n p hv e, n < nSeq d ∧ lo d n ≤ p ∧ p < hi d n ∧ hv < d.HV ∧ e < d.V ∧
    x = oIdx d p hv e}

/-- What the launch leaves: for every sequence `n`, token `p` of it, value head `hv < HV` and
column `e < V`, `dv[p, hv, e]` is a finite `bfloat16` within `tol` of `ref` over the initial
inputs; every byte outside the written elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ n p hv e, n < nSeq d → lo d n ≤ p → p < hi d n → hv < d.HV → e < d.V →
    ∃ c : BF16, rdB fin d.dv (oIdx d p hv e) = some c ∧ c.isFinite = true ∧
      |c.toReal - ref d init n p hv e| ≤ tol d init n p hv e) ∧
  ∀ z, z ∉ d.dv.elems (written d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `chunk_bwd_kernel_dv_local` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## The host's launches

`canon vl G B T H HV K V BT BK BV sc dt N NT` is the host's launch for those sizes: contiguous
`q`, `k` (`[B, T, H, K]`), `do`, `dv = empty_like(do)` (`[B, T, HV, V]`) and `g` (`[B, T, HV]`), the
storages at `2^40`, `2·2^40`, …, `7·2^40`; on a variable-length launch `N` sequences and `NT`
rows of `chunk_indices` (the tables' contents are `Pre`'s, which `check_gpu.py` checks on the
host's tables). -/
def canon (vl G : Bool) (B T H HV K V BT BK BV : Nat) (sc : F64) (dt : DType) (N NT : Nat)
    (cuv : Nat → dt.Val := fun _ => dt.ofBits 0) (civ : Nat → dt.Val := fun _ => dt.ofBits 0) :
    Inst where
  varlen := vl
  useG := G
  B := B
  T := T
  H := H
  HV := HV
  K := K
  V := V
  BT := BT
  BK := BK
  BV := BV
  scale := sc
  q := ⟨2 ^ 40, B * T * H * K⟩
  k := ⟨2 * 2 ^ 40, B * T * H * K⟩
  dO := ⟨3 * 2 ^ 40, B * T * HV * V⟩
  dv := ⟨4 * 2 ^ 40, B * T * HV * V⟩
  g := ⟨5 * 2 ^ 40, B * T * HV⟩
  idxD := dt
  N := N
  cu := ⟨6 * 2 ^ 40, N + 1⟩
  cuVal := cuv
  NT := NT
  ci := ⟨7 * 2 ^ 40, 2 * NT⟩
  ciVal := civ

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- **The host's launches are covered** at every size within the bounds: `chunk_size` 16, 32
or 64, the host's `BK`, `BV`, head dims `K ≤ 256`, grouped value heads, `B·HV < 2^16`, fewer than
`2^31` programs, a `scale` in `[2^-32, 2^32]`, `V ≤ 2^16`, `HV·V < 2^31`, at most `2^22` tokens,
at most `8192` rows or sequences, tensors of at most `2^38` elements and tables of `int32` or `int64`. -/
theorem canon_ok {vl G : Bool} {B T H HV K V BT BK BV : Nat} {sc : F64} {dt : DType} {N NT : Nat}
    {cuv civ : Nat → dt.Val}
    (hbt : BT = 16 ∨ BT = 32 ∨ BT = 64) (hbk : ∃ a, BK = 2 ^ a) (hbk1 : 16 ≤ BK) (hbk2 : BK ≤ 128)
    (hbv : ∃ a, BV = 2 ^ a) (hbv1 : 16 ≤ BV) (hbv2 : BV ≤ 128) (hK0 : 0 < K) (hK : K ≤ 256)
    (hV0 : 0 < V) (hH0 : 0 < H) (hHV0 : 0 < HV) (hdvd : H ∣ HV) (hB0 : 0 < B)
    (hvB : vl = true → B = 1) (hT : T < 2 ^ 31)
    (hg0 : nChunks (canon vl G B T H HV K V BT BK BV sc dt N NT cuv civ) < 2 ^ 31)
    (hg1 : B * HV < 2 ^ 16)
    (hg : nChunks (canon vl G B T H HV K V BT BK BV sc dt N NT cuv civ) * (B * HV) < 2 ^ 31)
    (hsc : sc.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ sc.toRat ∧ sc.toRat ≤ 2 ^ 32)
    (hq : B * T * H * K ≤ 2 ^ 38) (hd : B * T * HV * V ≤ 2 ^ 38)
    (hdt : dt = .i32 ∨ dt = .i64) (hN : N < 2 ^ 31) (hNT : 2 * NT ≤ 2 ^ 36)
    (hVle : V ≤ 2 ^ 16) (hHVV : HV * V < 2 ^ 31) (hTc : T ≤ 2 ^ 22) (hB13 : B ≤ 8192)
    (hN13 : N ≤ 8192) :
    (canon vl G B T H HV K V BT BK BV sc dt N NT cuv civ).ok := by
  have hbytes : dt.bytes ≤ 8 := by rcases hdt with h | h <;> subst h <;> decide
  have h1 : ((B * T * H * K : Nat) : Int) ≤ 2 ^ 38 := by exact_mod_cast hq
  have h2 : ((B * T * HV * V : Nat) : Int) ≤ 2 ^ 38 := by exact_mod_cast hd
  have h3 : ((N + 1 : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : (N + 1) * dt.bytes ≤ (N + 1) * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this (by omega : (N + 1) * 8 ≤ 2 ^ 39)
  have h4 : ((2 * NT : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : 2 * NT * dt.bytes ≤ 2 * NT * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this (by omega : 2 * NT * 8 ≤ 2 ^ 39)
  refine ⟨hbt, hbk, hbk1, hbk2, hbv, hbv1, hbv2, hK0, hK, hV0, hVle, hHVV, hH0, hHV0, hdvd,
    hB0, hvB, hT, hTc, hB13, hg0, hg1, hg, hsc, le_refl _, le_refl _, le_refl _, le_refl _, fun _ => le_refl _, ?_, ?_, ?_,
    fun _ => ?_, fun _ => hdt, fun _ => hN, fun _ => hN13, fun _ => le_refl _, fun _ => le_refl _, fun _ => ?_,
    fun _ => ?_⟩
  · exact bytes_disj _ _ (Or.inr (by simp only [canon, DType.bytes]; push_cast at h1 h2 ⊢; omega))
  · exact bytes_disj _ _ (Or.inr (by simp only [canon, DType.bytes]; push_cast at h1 h2 ⊢; omega))
  · exact bytes_disj _ _ (Or.inr (by simp only [canon, DType.bytes]; push_cast at h1 h2 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; push_cast at h1 h2 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; push_cast at h2 h3 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; push_cast at h2 h4 ⊢; omega))

/-! ## Coverage: SETUP.md's launches

The host's test cases (source lines 278–290), each without and with a gate (`G`); the code base's
benchmark shapes (`benchmarks/ops/registry.py:203-210`, `K = V = D`); gated DeltaNet's default
heads (`H = HV = 6`, `K = 256`, `V = 512`); a packed training batch of `2^16` tokens with
Qwen3-Next's grouped value heads (`H = 16`, `HV = 32`), with `int32` and `int64` tables; and the
largest `B·HV` the grid takes (`65535`). `scale = K ** -0.5` (the code base's default), the doubles
`s16` … `s256`. -/

/-- `K ** -0.5` for `K = 16, 32, 60, 64, 100, 128, 256`, as the host computes it (doubles). -/
def s16 : F64 := FloatV.ofPat 0x3fd0000000000000
def s32 : F64 := FloatV.ofPat 0x3fc6a09e667f3bcd
def s60 : F64 := FloatV.ofPat 0x3fc08654a2d4f6db
def s64 : F64 := FloatV.ofPat 0x3fc0000000000000
def s100 : F64 := FloatV.ofPat 0x3fb999999999999a
def s128 : F64 := FloatV.ofPat 0x3fb6a09e667f3bcd
def s256 : F64 := FloatV.ofPat 0x3fb0000000000000

theorem s16_ok : s16.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s16.toRat ∧ s16.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s32_ok : s32.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s32.toRat ∧ s32.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s60_ok : s60.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s60.toRat ∧ s60.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s64_ok : s64.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s64.toRat ∧ s64.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s100_ok : s100.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s100.toRat ∧ s100.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s128_ok : s128.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s128.toRat ∧ s128.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem s256_ok : s256.isFinite = true ∧ (1 / 2 ^ 32 : ℚ) ≤ s256.toRat ∧ s256.toRat ≤ 2 ^ 32 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- A fixed-length launch (`cu_seqlens = None`): `B` rows of `T` tokens. -/
abbrev padded (G : Bool) (B T H HV K V BT BK BV : Nat) (sc : F64) : Inst :=
  canon false G B T H HV K V BT BK BV sc .i64 0 0

/-- A variable-length launch with `N` sequences, `NT` rows and tables of dtype `dt`. -/
abbrev packed (G : Bool) (T H HV K V BT BK BV : Nat) (sc : F64) (dt : DType) (N NT : Nat) : Inst :=
  canon true G 1 T H HV K V BT BK BV sc dt N NT

/-- The host's cases 1–8 (fixed length) and 9–13 (variable length, `int64` tables, `NT` as
`prepare_chunk_indices` gives it). -/
def setup_t63 (G : Bool) : Inst := padded G 1 63 1 1 64 64 64 64 64 s64
def setup_t500 (G : Bool) : Inst := padded G 2 500 4 4 60 60 64 64 64 s60
def setup_t1000 (G : Bool) : Inst := padded G 2 1000 2 8 128 128 64 128 128 s128
def setup_t1024 (G : Bool) : Inst := padded G 3 1024 4 4 100 100 64 128 128 s100
def setup_bt16 (G : Bool) : Inst := padded G 1 64 2 4 32 32 16 32 32 s32
def setup_bt32 (G : Bool) : Inst := padded G 1 64 2 4 32 32 32 32 32 s32
def setup_gdn (G : Bool) : Inst := padded G 2 1024 6 6 256 512 64 128 128 s256
def setup_qwen (G : Bool) : Inst := padded G 1 2048 16 32 128 128 64 128 128 s128
def setup_var1 (G : Bool) : Inst := packed G 15 4 4 60 60 64 64 64 s60 .i64 1 1
/-- `setup_var3` carries its real tables: `cu_seqlens = [0, 256, 500, 1000]` and the 16 rows
`prepare_chunk_indices` builds (`setup_var3_tables`). -/
def var3cu : List Nat := [0, 256, 500, 1000]
def var3ci : List Nat :=
  [0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7]
def setup_var3 (G : Bool) : Inst :=
  canon true G 1 1000 4 4 64 64 64 64 64 s64 .i64 3 16
    (fun i => DType.ofBits .i64 (var3cu.getD i 0)) (fun i => DType.ofBits .i64 (var3ci.getD i 0))
def setup_var5 (G : Bool) : Inst := packed G 2000 4 4 100 100 64 128 128 s100 .i64 5 35
def setup_var3g (G : Bool) : Inst := packed G 1000 2 8 64 64 64 64 64 s64 .i64 3 16
def setup_var6 (G : Bool) : Inst := packed G 4096 4 8 64 64 64 64 64 s64 .i64 6 68

/-- The benchmark shapes (`B, T, H, D`). -/
def setup_b1 (G : Bool) : Inst := padded G 1 8192 96 96 128 128 64 128 128 s128
def setup_b2 (G : Bool) : Inst := padded G 2 16384 16 16 128 128 64 128 128 s128
def setup_b3 (G : Bool) : Inst := padded G 4 2048 16 16 128 128 64 128 128 s128
def setup_b4 (G : Bool) : Inst := padded G 4 4096 64 64 128 128 64 128 128 s128
def setup_b5 (G : Bool) : Inst := padded G 8 2048 32 32 256 256 64 128 128 s256
def setup_b6 (G : Bool) : Inst := padded G 8 1024 8 8 64 64 64 64 64 s64

/-- Gated DeltaNet's default heads at its default context (2048 tokens), eight rows. -/
def setup_gdn_default (G : Bool) : Inst := padded G 8 2048 6 6 256 512 64 128 128 s256

/-- A packed batch of `2^16` tokens, lengths `[1, 63, 1, 4031, 25904, 35536]`, Qwen3-Next's heads,
with `int32` (`dt = .i32`) or `int64` tables. -/
def setup_train (G : Bool) (dt : DType) : Inst := packed G 65536 16 32 128 128 64 128 128 s128 dt 6 1027

/-- The largest grid `y` the kernel launches: `B·HV = 65535` (`B = 4369`, `HV = 15`). -/
def setup_rows (G : Bool) : Inst := padded G 4369 16 15 15 16 16 16 16 16 s16

theorem pow7 : (128 : Nat) = 2 ^ 7 := rfl
theorem pow6 : (64 : Nat) = 2 ^ 6 := rfl
theorem pow5 : (32 : Nat) = 2 ^ 5 := rfl
theorem pow4 : (16 : Nat) = 2 ^ 4 := rfl

theorem setup_t63_ok (G : Bool) : (setup_t63 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_t500_ok (G : Bool) : (setup_t500 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s60_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_t1000_ok (G : Bool) : (setup_t1000 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_t1024_ok (G : Bool) : (setup_t1024 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s100_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_bt16_ok (G : Bool) : (setup_bt16 G).ok :=
  canon_ok (by norm_num) ⟨5, pow5⟩ (by norm_num) (by norm_num) ⟨5, pow5⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s32_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_bt32_ok (G : Bool) : (setup_bt32 G).ok :=
  canon_ok (by norm_num) ⟨5, pow5⟩ (by norm_num) (by norm_num) ⟨5, pow5⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s32_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_gdn_ok (G : Bool) : (setup_gdn G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s256_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_qwen_ok (G : Bool) : (setup_qwen G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_var1_ok (G : Bool) : (setup_var1 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s60_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_var3_ok (G : Bool) : (setup_var3 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- `setup_var3`'s real tables are `TablesOk` (`Pre`'s table clause on the host's tables). -/
theorem setup_var3_tables (G : Bool) : TablesOk (setup_var3 G) := by
  have hcu : ∀ i, i ≤ 3 → cuAt (setup_var3 G) i = (var3cu.getD i 0 : Int) := by
    intro i hi; interval_cases i <;> (cases G <;> decide)
  have hseq : ∀ r, r < 16 → seqAt (setup_var3 G) r = (var3ci.getD (2 * r) 0 : Int) := by
    intro r hr; interval_cases r <;> (cases G <;> decide)
  have hchk : ∀ r, r < 16 → chkAt (setup_var3 G) r = (var3ci.getD (2 * r + 1) 0 : Int) := by
    intro r hr; interval_cases r <;> (cases G <;> decide)
  have hN : (setup_var3 G).N = 3 := rfl
  have hNT : (setup_var3 G).NT = 16 := rfl
  have hT : (setup_var3 G).T = 1000 := rfl
  have hBT : (setup_var3 G).BT = 64 := rfl
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hcu 0 (by norm_num)]; (cases G <;> decide)
  · intro i hi; rw [hN] at hi
    rw [hcu i (by omega), hcu (i + 1) (by omega)]; interval_cases i <;> (cases G <;> decide)
  · rw [hN, hT, hcu 3 le_rfl]; (cases G <;> decide)
  · intro r hr; rw [hNT] at hr
    interval_cases r <;> (cases G <;> decide)
  · intro r r' hr hr' hne; rw [hNT] at hr hr'
    rw [hseq r hr, hchk r hr, hseq r' hr', hchk r' hr']
    have key : ∀ r, r < 16 → ∀ r', r' < 16 → r ≠ r' →
        (var3ci.getD (2 * r) 0, var3ci.getD (2 * r + 1) 0) ≠
          (var3ci.getD (2 * r') 0, var3ci.getD (2 * r' + 1) 0) := by decide
    intro h
    apply key r hr r' hr' hne
    simp only [Prod.mk.injEq, Nat.cast_inj] at h ⊢
    exact h
  · intro n c hn hc; rw [hN] at hn; rw [hBT] at hc
    rw [hcu (n + 1) (by omega), hcu n (by omega)] at hc
    rw [hNT]
    interval_cases n
    · have : c < 4 := by simp [var3cu] at hc; omega
      interval_cases c
      · exact ⟨0, by norm_num, by rw [hseq 0 (by norm_num)]; rfl, by rw [hchk 0 (by norm_num)]; rfl⟩
      · exact ⟨1, by norm_num, by rw [hseq 1 (by norm_num)]; rfl, by rw [hchk 1 (by norm_num)]; rfl⟩
      · exact ⟨2, by norm_num, by rw [hseq 2 (by norm_num)]; rfl, by rw [hchk 2 (by norm_num)]; rfl⟩
      · exact ⟨3, by norm_num, by rw [hseq 3 (by norm_num)]; rfl, by rw [hchk 3 (by norm_num)]; rfl⟩
    · have : c < 4 := by simp [var3cu] at hc; omega
      interval_cases c
      · exact ⟨4, by norm_num, by rw [hseq 4 (by norm_num)]; rfl, by rw [hchk 4 (by norm_num)]; rfl⟩
      · exact ⟨5, by norm_num, by rw [hseq 5 (by norm_num)]; rfl, by rw [hchk 5 (by norm_num)]; rfl⟩
      · exact ⟨6, by norm_num, by rw [hseq 6 (by norm_num)]; rfl, by rw [hchk 6 (by norm_num)]; rfl⟩
      · exact ⟨7, by norm_num, by rw [hseq 7 (by norm_num)]; rfl, by rw [hchk 7 (by norm_num)]; rfl⟩
    · have : c < 8 := by simp [var3cu] at hc; omega
      interval_cases c
      · exact ⟨8, by norm_num, by rw [hseq 8 (by norm_num)]; rfl, by rw [hchk 8 (by norm_num)]; rfl⟩
      · exact ⟨9, by norm_num, by rw [hseq 9 (by norm_num)]; rfl, by rw [hchk 9 (by norm_num)]; rfl⟩
      · exact ⟨10, by norm_num, by rw [hseq 10 (by norm_num)]; rfl, by rw [hchk 10 (by norm_num)]; rfl⟩
      · exact ⟨11, by norm_num, by rw [hseq 11 (by norm_num)]; rfl, by rw [hchk 11 (by norm_num)]; rfl⟩
      · exact ⟨12, by norm_num, by rw [hseq 12 (by norm_num)]; rfl, by rw [hchk 12 (by norm_num)]; rfl⟩
      · exact ⟨13, by norm_num, by rw [hseq 13 (by norm_num)]; rfl, by rw [hchk 13 (by norm_num)]; rfl⟩
      · exact ⟨14, by norm_num, by rw [hseq 14 (by norm_num)]; rfl, by rw [hchk 14 (by norm_num)]; rfl⟩
      · exact ⟨15, by norm_num, by rw [hseq 15 (by norm_num)]; rfl, by rw [hchk 15 (by norm_num)]; rfl⟩

theorem setup_var5_ok (G : Bool) : (setup_var5 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s100_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_var3g_ok (G : Bool) : (setup_var3g G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_var6_ok (G : Bool) : (setup_var6 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b1_ok (G : Bool) : (setup_b1 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b2_ok (G : Bool) : (setup_b2 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b3_ok (G : Bool) : (setup_b3 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b4_ok (G : Bool) : (setup_b4 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b5_ok (G : Bool) : (setup_b5 G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s256_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_b6_ok (G : Bool) : (setup_b6 G).ok :=
  canon_ok (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num) ⟨6, pow6⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_gdn_default_ok (G : Bool) : (setup_gdn_default G).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s256_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_train_ok (G : Bool) {dt : DType} (hdt : dt = .i32 ∨ dt = .i64) :
    (setup_train G dt).ok :=
  canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s128_ok (by norm_num) (by norm_num) hdt
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
theorem setup_rows_ok (G : Bool) : (setup_rows G).ok :=
  canon_ok (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s16_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

/-! ## Satisfiable

One row of one token, one head, head dims 16 (`BK = BV = 16`, the smallest that compiles:
`tl.dot` needs `K ≥ 16`), `chunk_size = 16`, no gate, every byte `0`. -/

def satInst : Inst := padded false 1 1 1 1 16 16 16 16 16 s16

theorem satInst_ok : satInst.ok :=
  canon_ok (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s16_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem zeroMem_readBytes (a : Int) : ∀ n, zeroMem.readBytes a n = some (List.replicate n 0)
  | 0 => rfl
  | n + 1 => by simp [Mem.readBytes, zeroMem_readBytes (a + 1) n, zeroMem, List.replicate_succ]

theorem leValue_zeros : ∀ n, TVA.Bytes.leValue (List.replicate n (0 : Fin 256)) = 0
  | 0 => rfl
  | n + 1 => by simp [List.replicate_succ, TVA.Bytes.leValue, leValue_zeros n]

/-- Every element of a `bfloat16` buffer at an address inside `[0, 2^63]` reads `+0`. -/
theorem zero_rdB (b : Buf .bf16) (i : Nat) (h0 : 0 ≤ b.base) (h1 : b.base + i * 2 + 2 ≤ 2 ^ 64)
    (h2 : b.base % 2 = 0) : rdB zeroMem b i = some (FloatV.ofPat 0) := by
  have hal : Mem.Aligned .bf16 (b.ptr (i : Int)).addr := by
    refine ⟨?_, ?_, ?_⟩
    · simp only [Buf.ptr_addr, DType.bytes]; positivity
    · simp only [Buf.ptr_addr, DType.bytes]; omega
    · simp only [Buf.ptr_addr, DType.bytes]; omega
  simp only [rdB, Mem.read, hal, ite_true, zeroMem_readBytes, Option.map_some, leValue_zeros]
  rfl

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨satInst, zeroMem, satInst_ok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid zeroMem := fun b h1 h2 h3 =>
    ⟨h1, h2, h3, fun x _ => by simp [zeroMem]⟩
  have hz : ∀ (b : Buf .bf16), b = satInst.q ∨ b = satInst.k ∨ b = satInst.dO →
      ∀ i, i < 16 → valB zeroMem b i = 0 ∧ finB zeroMem b i := by
    intro b hb i hi
    have hr : rdB zeroMem b i = some (FloatV.ofPat 0) := by
      refine zero_rdB b i ?_ ?_ ?_ <;>
        rcases hb with rfl | rfl | rfl <;> simp [satInst, canon] <;> omega
    exact ⟨by simp only [valB, hr]; decide +kernel, ⟨_, hr, by decide⟩⟩
  refine ⟨hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    fun h => by simp [satInst, canon] at h, fun h => by simp [satInst, canon] at h, ?_⟩
  intro n hn
  have hn0 : n = 0 := by simp [nSeq, satInst, canon] at hn; omega
  subst hn0
  have hlo : lo satInst 0 = 0 := by simp [lo, satInst, canon]
  have hhi : hi satInst 0 = 1 := by simp [hi, satInst, canon]
  have hend : ∀ p, p < 1 → chunkEnd satInst 0 p = 1 := by
    intro p hp
    have : p = 0 := by omega
    subst this
    simp only [chunkEnd, chunkStart, hlo, hhi]
    simp [satInst, canon]
  have hmag : ∀ p j hv, p < 1 → j < 1 → hv < 1 → kqMag satInst zeroMem p j hv = 0 := by
    intro p j hv hp hj hhv
    have hp0 : p = 0 := by omega
    have hj0 : j = 0 := by omega
    have hv0 : hv = 0 := by omega
    subst hp0 hj0 hv0
    unfold kqMag
    refine Finset.sum_eq_zero fun c hc => ?_
    have hc' : c < 16 := by simpa [satInst, canon] using hc
    have hidx : qkIdx satInst 0 (hk satInst 0) c = c := by simp [qkIdx, hk, satInst, canon]
    rw [hidx, (hz _ (Or.inr (Or.inl rfl)) c hc').1, (hz _ (Or.inl rfl) c hc').1]
    simp [Device.liftMag]
  refine ⟨fun p hp1 hp2 => ?_, fun p j hv hp hpj hj hhv => ?_, fun p hv e hp1 hp2 hhv he => ?_⟩
  · rw [hlo] at hp1; rw [hhi] at hp2
    have hp0 : p = 0 := by omega
    subst hp0
    refine ⟨fun h c hh hc => ?_, fun hv e hhv he => ?_, fun h => by simp [satInst, canon] at h⟩
    · have hh0 : h = 0 := by simpa [satInst, canon] using hh
      have hc' : c < 16 := by simpa [satInst, canon] using hc
      subst hh0
      have hidx : qkIdx satInst 0 0 c = c := by simp [qkIdx, satInst, canon]
      rw [hidx]
      exact ⟨(hz _ (Or.inl rfl) c hc').2, (hz _ (Or.inr (Or.inl rfl)) c hc').2⟩
    · have hv0 : hv = 0 := by simpa [satInst, canon] using hhv
      have he' : e < 16 := by simpa [satInst, canon] using he
      subst hv0
      have hidx : oIdx satInst 0 0 e = e := by simp [oIdx, satInst, canon]
      rw [hidx, (hz _ (Or.inr (Or.inr rfl)) e he').1]
      exact ⟨(hz _ (Or.inr (Or.inr rfl)) e he').2, by norm_num⟩
  · rw [hlo] at hp
    have hce : chunkEnd satInst 0 p ≤ 1 := by rw [← hhi]; exact min_le_right _ _
    have hj1 : j < 1 := by omega
    have hp1 : p < 1 := by omega
    have hv1 : hv < 1 := by simpa [satInst, canon] using hhv
    rw [hmag p j hv hp1 hj1 hv1]
    exact ⟨fun h => by simp [satInst, canon] at h, by norm_num, by norm_num⟩
  · rw [hlo] at hp1; rw [hhi] at hp2
    have hv1 : hv < 1 := by simpa [satInst, canon] using hhv
    rw [hend p hp2]
    refine le_trans (le_of_eq (Finset.sum_eq_zero fun j hj => ?_)) (by norm_num)
    have hj1 : j < 1 := (Finset.mem_Ico.mp hj).2
    rw [hmag p j hv hp2 hj1 hv1]; ring

end Claims.ChunkBwdDvLocal
