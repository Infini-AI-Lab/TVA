import TVA

/-!
# `mean_pooling_fwd_bf16`: the mean over each chunk of tokens, fla

The kernel is `mean_pooling_fwd_kernel` of `source/pooling.py` (lines 81–117), which calls the
`@triton.jit` helper `unflatten_program_id` (lines 71–78); it reads the module-level names `triton`
and `tl` (lines 67–68). The entry point `mean_pooling` (lines 191–211) launches it through
`mean_pooling_fwd` (lines 159–188) on a `bfloat16` input `x` (`[B, T, H, D]`) and a fresh
`bfloat16` output `o` (`[B, NT, H, D]`), over fixed-length rows (`cu_seqlens is None`) or a packed
variable-length batch (`B = 1`, `cu_seqlens` and `chunk_indices` of one dtype, `int32` or
`int64`); one program per (feature block, chunk) and (batch row, head). `SETUP.md` lists the
launches.

What is claimed: every covered launch ends normally; for every output row `g` (chunk `c` of row
`b`, `g = b·NT + c`, on a fixed-length launch; the chunk named by row `g` of `chunk_indices` on a
variable-length one), every head `h` and every feature `j < D`, the output element `o[g, h, j]`
holds a `bfloat16` value which, when the chunk's inputs are finite and their magnitudes leave room
(`InRange`), is finite and within `tol` of the exact mean `μ = (1/L)·Σ x[p, h, j]` over the chunk's
`L` positions `p` (chunks of `BT` positions counted from the sequence's start, the last one
possibly shorter); every other byte is unchanged.

The conditions: `BT` is a power of two, `BD` a power of two in `[16, 128]` (the source's autotune
list), `BT·BD ≤ 2^20`; `B, H, D ≥ 1`, `T < 2^31`, `T ≤ 2^22`, `B ≤ 8192`, `D ≤ 2^17`,
`B·H < 2^16` (the grid's axis 1); the grid has fewer than `2^31` programs; `x` holds `B·T·H·D`
elements, `o` holds every output row and shares no byte with `x`. On a variable-length launch:
`B = 1`, `int32` or `int64` tables, `N < 2^31` and `N ≤ 8192` sequences, tables long enough and
disjoint from `o`, and their contents well formed (`TablesOk`). Every `bfloat16` input is covered.

The float operations are `tl.sum`'s IEEE additions, the platform's `float32` division
(`Device.DivFull`), an exact integer-to-float conversion and the rounding to `bfloat16`; the
tolerance also uses the depth of a reduction's tree (`Device.SumDepth`).
-/

namespace Claims.MeanPoolingFwd

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `mean_pooling_fwd_kernel` (source lines 81–117) with `unflatten_program_id` (lines 71–78)
inlined at line 94 (`Y` its default `None`: the compile-time `if Y is None` takes the
`pid % X, pid // X` branch). The pointers `x`, `o` are `*bf16`; `cu_seqlens` and `chunk_indices`
are pointers of the table dtype `idxD` (`int32` or `int64`; `None` on a fixed-length launch, never
read); `T` is the runtime `int32` (`do_not_specialize=['T']`: never a compile-time constant). `H`,
`D`, `BT`, `BD` and `IS_VARLEN` are `tl.constexpr`, Lean parameters; the flag's branches are
compile-time `if`s, written as Lean `if`s. -/
def mean_pooling_fwd_kernel (x_ptr o_ptr : Ptr .bf16) (idxD : DType) (cu_ptr ci_ptr : Ptr idxD)
    (T : I32) (H D BT BD : Nat) (IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 94 (lines 74 and 76): `pid = tl.program_id(0).to(tl.int64)`, then `pid % X, pid // X`
    -- with `X = tl.cdiv(D, BD)`, Python's ceiling division of the constexprs (`(D + BD - 1) // BD`
    -- on non-negative integers), built in `int64`
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    let i_d := (← tl.irem (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat ((D + BD - 1) / BD) : I64))).item
    let i_t := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat ((D + BD - 1) / BD) : I64))).item
    -- line 95: axis 1 of the grid, `.to(tl.int64)`
    let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
    -- line 96: `int64` `//` and `%` by the constexpr `H`
    let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat H : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat H : I64))).item
    -- lines 97–106: the compile-time `if IS_VARLEN:`; each branch gives the output row `i_tg`, the
    -- chunk index `i_t`, `bos`, and the `T` that lines 110 and 116 read, as an `int64` (on the
    -- fixed-length branch `T` is the `int32` argument, which both uses promote to `int64`)
    let (i_tg, i_t, bos, T64) ← if IS_VARLEN then do
        -- line 98
        let i_tg := i_t
        -- line 99: both loads use the old `i_t`; `chunk_indices[i_t, 0]` `.to(tl.int32)` (of an
        -- `int32` value the identity), `chunk_indices[i_t, 1]` `.to(tl.int64)`; the `int64`
        -- `i_t * 2`, and `+ 1` (the Python `1`, an `int32`) added to the pointer
        let i_n := (← tl.convert idxD .i32 (Tensor.scalar
          (← tl.load1 (ci_ptr + i_t * (IntV.ofNat 2 : I64))))).item
        let i_t := (← tl.convert idxD .i64 (Tensor.scalar
          (← tl.load1 (ci_ptr + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
        -- line 100: the sequence's bounds, `.to(tl.int64)`
        let bos := (← tl.convert idxD .i64 (Tensor.scalar (← tl.load1 (cu_ptr + i_n)))).item
        let eos := (← tl.convert idxD .i64
          (Tensor.scalar (← tl.load1 (cu_ptr + i_n + (IntV.ofNat 1 : I32))))).item
        -- line 101: `T = eos - bos`, the wrapping `int64` subtraction
        let T := eos - bos
        -- line 102: `tl.cdiv(T, BT)` in `int64` (the constexpr built in `int64`); not used
        let _NT ← tl.cdiv (Tensor.scalar T) (Tensor.scalar (IntV.ofNat BT : I64))
        pure (i_tg, i_t, bos, T)
      else do
        -- line 104: `tl.cdiv(T, BT)` of the `int32` `T` (the constexpr built in `int32`)
        let NT := (← tl.cdiv (Tensor.scalar T) (Tensor.scalar (IntV.ofNat BT : I32))).item
        -- line 105: the `int64` `i_b` times `NT` (promoted to `int64`), plus `i_t`
        let NT64 := (← tl.convert .i32 .i64 (Tensor.scalar NT)).item
        let i_tg := i_b * NT64 + i_t
        -- line 106: `bos = i_b * T`, the `int32` `T` promoted to `int64`; `eos` is not used on
        -- this path
        let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
        pure (i_tg, i_t, i_b * T64, T64)
    -- line 108: the `int64` `i_t * BT` (the constexpr built in `int64`) plus `tl.arange(0, BT)`
    -- (`int32`, promoted to `int64`)
    let o_t := bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    -- line 109: the same for the feature block
    let o_d := bc (i_d * (IntV.ofNat BD : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BD))
    -- line 110: `o_t[:, None] < T` and `o_d[None, :] < D` (the Python `D` an `int32`, promoted to
    -- `int64`), signed `int64` comparisons, broadcast against each other to `[BT, BD]`; the model
    -- broadcasts the operands first, which gives the same lanes
    let m_x := ((o_t.col BD).ltS T64).and ((o_d.row BT).ltS (IntV.ofNat D : I64))
    -- line 111: left to right, `int64` products and sums (each constexpr, and the Python product
    -- `H*D`, built in `int64`), the pointer advanced in elements by the scalar, then by the
    -- `[BT, 1]` row term, then by the `[1, BD]` column term (broadcast to `[BT, BD]`)
    let p_x := x_ptr + (bos * (IntV.ofNat H : I64) + i_h) * (IntV.ofNat D : I64)
      + o_t.col BD * (IntV.ofNat (H * D) : I64) + o_d.row BT
    -- line 112
    let p_o := o_ptr + (i_tg * (IntV.ofNat H : I64) + i_h) * (IntV.ofNat D : I64) + o_d
    -- line 114: the masked load, `other=0.0` the `float32` `+0` converted to `bfloat16`; then
    -- `.to(tl.float32)` (exact)
    let b_x : Tensor [BT, BD] F32 ← tl.cvt (← tl.load p_x (some m_x)
      (some (← tl.convert .f32 .bf16 (tl.full [BT, BD] (FloatV.litF32 (FloatV.ofPat 0))))))
    -- line 116: `tl.sum(b_x, axis=0)`, IEEE additions along a tree per column (no product feeds
    -- it); then Python's `min` with a tensor operand, `tl.minimum`, on `int64` (the constexpr `BT`
    -- an `int32` tensor, promoted; `T - i_t * BT` the wrapping `int64` difference): the exact
    -- `tl.iminimum`; `/` of the `float32` block by the `int64` scalar converts the scalar to
    -- `float32` (`tl.itof`), broadcasts it, and divides with the platform's division
    let b_s ← tl.sum0 b_x
    let L := (tl.iminimum (Tensor.scalar (IntV.ofNat BT : I64))
      (Tensor.scalar (T64 - i_t * (IntV.ofNat BT : I64)))).item
    let L32 : Tensor [] F32 ← tl.itof (Tensor.scalar L)
    let b_o ← tl.div b_s (bc L32.item)
    -- line 117: `.to(p_o.dtype.element_ty)`, one rounding to nearest even into `bfloat16`; the
    -- masked store, `o_d < D` as on line 110
    let out : Tensor [BD] BF16 ← tl.cvt b_o
    tl.store p_o out (some (o_d.ltS (IntV.ofNat D : I64)))

/-! ## The conditions -/

/-- What the host chooses: one launch through `mean_pooling(x, chunk_size, cu_seqlens)` (or
`mean_pooling_fwd(x, chunk_size, cu_seqlens, chunk_indices)`) on a 4-D `bfloat16` `x`, with the
contents of its input tensors. -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `B = x.shape[0]` (`1` on a variable-length launch). -/
  B : Nat
  /-- `T = x.shape[1]`. -/
  T : Nat
  /-- `H = x.shape[2]`. -/
  H : Nat
  /-- `D = x.shape[3]`. -/
  D : Nat
  /-- `BT = chunk_size`. -/
  BT : Nat
  /-- `BD`, the feature block (the host's `64`, one configuration of the source's autotune list). -/
  BD : Nat
  /-- `x`, contiguous `[B, T, H, D]` `bfloat16`: element `(b, t, h, j)` is `((b·T + t)·H + h)·D + j`. -/
  x : Buf .bf16
  /-- Its contents. -/
  xv : Nat → BF16
  /-- `o`, fresh, contiguous `[B, NT, H, D]` `bfloat16`: element `(g, h, j)` of the flattened
  `[B·NT, H, D]` is `(g·H + h)·D + j`. -/
  o : Buf .bf16
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

/-- The chunks of a fixed-length row: `triton.cdiv(T, BT)`. -/
def ntFix (d : Inst) : Nat := (d.T + d.BT - 1) / d.BT

/-- The number of output rows (`B·NT`, the output's first two axes): `NT = len(chunk_indices)`
(variable length, `B = 1`) or `B·cdiv(T, BT)` (fixed length). -/
def nRows (d : Inst) : Nat := if d.varlen then d.NT else d.B * ntFix d

/-- Axis 0 of the grid: `triton.cdiv(D, BD) · NT`. -/
def gridX (d : Inst) : Nat := (d.D + d.BD - 1) / d.BD * (if d.varlen then d.NT else ntFix d)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BT` a power of two (`tl.arange`). -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  /-- `BD` a power of two in `[16, 128]`: the values of the source's autotune list. -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  bd_ge : 16 ≤ d.BD
  bd_le : d.BD ≤ 128
  /-- The `[BT, BD]` block has at most `2^20` elements (Triton's largest block). -/
  block_le : d.BT * d.BD ≤ 2 ^ 20
  /-- At least one batch row, head and feature. -/
  B_pos : 0 < d.B
  H_pos : 0 < d.H
  D_pos : 0 < d.D
  /-- `T` is an `int32` (`do_not_specialize`: a runtime `int32` below `2^31`). -/
  T_lt : d.T < 2 ^ 31
  /-- `T` is a context length (the tokens of a row, or of the packed batch): at most `2^22`. -/
  T_le : d.T ≤ 2 ^ 22
  /-- `B`, the batch rows, is a batch size: at most `8192`. -/
  B_le : d.B ≤ 8192
  /-- `D`, the features of a head, is a model dimension: at most `2^17`. -/
  D_le : d.D ≤ 2 ^ 17
  /-- `B·H < 2^16`: CUDA's limit on the grid's axis 1, `B * H` (README, Exclusions). -/
  BH_lt : d.B * d.H < 2 ^ 16
  /-- The grid has fewer than `2^31` programs (README, Exclusions). -/
  grid_lt : gridX d * (d.B * d.H) < 2 ^ 31
  /-- Variable length: `B = 1` (the entry point checks it). -/
  varlen_B : d.varlen = true → d.B = 1
  /-- The buffers hold the tensors. -/
  x_len : d.B * d.T * d.H * d.D ≤ d.x.len
  o_len : nRows d * d.H * d.D ≤ d.o.len
  /-- `o` is a fresh allocation: it shares no byte with `x`. -/
  o_x : Disjoint d.o.bytes d.x.bytes
  /-- Variable length: `int32` or `int64` tables, `N < 2^31` (a sequence index is an `int32`
  after line 99's `.to(tl.int32)`), the tables' lengths, and `o` sharing no byte with them. -/
  idx_ok : d.varlen = true → d.idxD = .i32 ∨ d.idxD = .i64
  N_lt : d.varlen = true → d.N < 2 ^ 31
  /-- Variable length: `N`, the number of sequences, is a batch size: at most `8192`. -/
  N_le : d.varlen = true → d.N ≤ 8192
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  ci_len : d.varlen = true → 2 * d.NT ≤ d.ci.len
  o_cu : d.varlen = true → Disjoint d.o.bytes d.cu.bytes
  o_ci : d.varlen = true → Disjoint d.o.bytes d.ci.bytes

/-- The tables of a variable-length launch: `cu_seqlens` nonnegative, nondecreasing and at most
`T`; each row of `chunk_indices` names a sequence `n < N` and a chunk `c ≥ 0` that starts inside
it (`c·BT < cu_seqlens[n+1] − cu_seqlens[n]`). `prepare_chunk_indices` builds such a table, one row
per chunk of the batch in order; any such table is covered. -/
def TablesOk (d : Inst) : Prop :=
  0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧ cuAt d d.N ≤ d.T ∧
  ∀ r, r < d.NT → 0 ≤ seqAt d r ∧ seqAt d r < d.N ∧ 0 ≤ chkAt d r ∧
    chkAt d r * d.BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat

/-- The inputs covered: the buffers are valid and hold the instance's contents; on a
variable-length launch, the tables are `TablesOk`. Every `bfloat16` value of `x` is covered. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.o.Valid init ∧ d.x.Holds init d.xv ∧
    (d.varlen = true → d.cu.Valid init ∧ d.ci.Valid init ∧ d.cu.Holds init d.cuVal ∧
      d.ci.Holds init d.ciVal ∧ TablesOk d)

/-- The model on an instance. -/
def model (d : Inst) : Kernel :=
  mean_pooling_fwd_kernel (d.x.ptr 0) (d.o.ptr 0) d.idxD (d.cu.ptr 0) (d.ci.ptr 0)
    (IntV.ofNat d.T) d.H d.D d.BT d.BD d.varlen

/-- The host's launch (lines 172–187): the grid `(cdiv(D, BD)·NT, B·H)`, a 2-tuple, so axis 2 has
size `1`; the block shapes `[BT]` (`tl.arange(0, BT)`, `o_t`), `[BD]` (`o_d`, `p_o`, `b_o`),
`[BT, 1]` (`o_t[:, None]` and what is built on it), `[1, BD]` (`o_d[None, :]`) and `[BT, BD]`
(`m_x`, `p_x`, `b_x`); `enable_fp_fusion = true` (the default), `debug = false`, `num_ctas = 1`,
no `launch_pdl`. `num_warps = 2` is performance-only; `num_stages = 3` changes no outcome of a model
without a conflict inside a program. -/
def launch (d : Inst) : Launch where
  grid := ![gridX d, d.B * d.H, 1]
  fusion := true
  blocks := [[d.BT], [d.BD], [d.BT, 1], [1, d.BD], [d.BT, d.BD]]

/-! ## The specification -/

/-- The positions of output row `g` are `[start g, stop g)` of the flat time axis (`b·T + t` for
row `b` of a fixed-length batch): on a variable-length launch, chunk `c = chunk_indices[g, 1]` of
sequence `n = chunk_indices[g, 0]`, `[cu_seqlens[n] + c·BT, min(cu_seqlens[n] + c·BT + BT,
cu_seqlens[n+1]))`; on a fixed-length one, chunk `c = g mod cdiv(T, BT)` of row
`b = g div cdiv(T, BT)`, `[b·T + c·BT, min(b·T + c·BT + BT, b·T + T))`. -/
def start (d : Inst) (g : Nat) : Nat :=
  if d.varlen then (cuAt d (seqAt d g).toNat).toNat + (chkAt d g).toNat * d.BT
  else g / ntFix d * d.T + g % ntFix d * d.BT
def stop (d : Inst) (g : Nat) : Nat :=
  if d.varlen then min (start d g + d.BT) (cuAt d ((seqAt d g).toNat + 1)).toNat
  else min (start d g + d.BT) (g / ntFix d * d.T + d.T)

/-- The number of positions of output row `g`, `L` (the divisor, `min(BT, T − c·BT)`). -/
def len (d : Inst) (g : Nat) : Nat := stop d g - start d g

/-- `x[p, h, j]` (element `(p·H + h)·D + j`, `p` on the flat time axis). -/
def xAt (d : Inst) (p h j : Nat) : BF16 := d.xv ((p * d.H + h) * d.D + j)

/-- The exact sum over output row `g`'s positions at `(h, j)`, and the sum of the magnitudes. -/
def refSum (d : Inst) (g h j : Nat) : ℚ := ∑ p ∈ Finset.Ico (start d g) (stop d g), (xAt d p h j).toRat
def absSum (d : Inst) (g h j : Nat) : ℚ := ∑ p ∈ Finset.Ico (start d g) (stop d g), |(xAt d p h j).toRat|

/-- The exact mean `μ = refSum / L`, and the mean magnitude `Ā = absSum / L`. -/
def mean (d : Inst) (g h j : Nat) : ℚ := refSum d g h j / (len d g : ℚ)
def absMean (d : Inst) (g h j : Nat) : ℚ := absSum d g h j / (len d g : ℚ)

/-- The element is in range: its terms are finite and `Σ|x| ≤ 2^125` (`float32` overflows near
`2^128`, `Device.DivFull` holds up to quotients of `2^126`). -/
def InRange (d : Inst) (g h j : Nat) : Prop :=
  (∀ p ∈ Finset.Ico (start d g) (stop d g), (xAt d p h j).isFinite = true) ∧
    absSum d g h j ≤ 2 ^ 125

/-- **The tolerance** of an element with mean `μ` and mean magnitude `Ā`, chunks of `BT = 2^e`:
one `bfloat16` rounding of the result, `2^-8·|μ|`; `(e + 5)·2^-24·Ā` for the `float32` sum along a
tree of depth `e`, the division and their effect on the rounding; and `2^-132` for the formats'
subnormal roundings (README, Tolerances). -/
def tol (BT : Nat) (μ Ā : ℚ) : ℚ :=
  2 ^ (-8 : ℤ) * |μ| + ((Nat.log 2 BT : ℚ) + 5) * 2 ^ (-24 : ℤ) * Ā + 2 ^ (-132 : ℤ)

/-- What the launch leaves: for every output row `g`, head `h` and feature `j`, `o[g, h, j]` holds
a `bfloat16` value which, when the element is `InRange`, is finite and within `tol` of the mean;
every byte outside the output's `nRows·H·D` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ g h j, g < nRows d → h < d.H → j < d.D → ∃ c : BF16,
      fin.read .bf16 (d.o.ptr ((g * d.H + h) * d.D + j)).addr = some c ∧
      (InRange d g h j → c.isFinite = true ∧
        |c.toRat - mean d g h j| ≤ tol d.BT (mean d g h j) (absMean d g h j))) ∧
    ∀ z, z ∉ d.o.elems {e | e < nRows d * d.H * d.D} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `mean_pooling_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- An instance of the host's launch (`BD = 64`) with `x` at `2^32`, `o` at `2^37`, `cu_seqlens`
at `2^40`, `chunk_indices` at `2^41`, zero contents: `vl` variable length, `B`, `T`, `H`, `D`,
`BT = 2^e`, the tables' dtype, `N` sequences and `NT` rows. -/
def mk (vl : Bool) (B T H D e : Nat) (dt : DType) (N NT : Nat) : Inst where
  varlen := vl
  B := B
  T := T
  H := H
  D := D
  BT := 2 ^ e
  BD := 64
  x := ⟨2 ^ 32, B * T * H * D⟩
  xv := fun _ => FloatV.ofPat 0
  o := ⟨2 ^ 37, (if vl then NT else B * ((T + 2 ^ e - 1) / 2 ^ e)) * H * D⟩
  idxD := dt
  N := N
  cu := ⟨2 ^ 40, N + 1⟩
  cuVal := fun _ => dt.ofBits 0
  NT := NT
  ci := ⟨2 ^ 41, 2 * NT⟩
  ciVal := fun _ => dt.ofBits 0

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {vl : Bool} {B T H D e : Nat} {dt : DType} {N NT : Nat}
    (he : e ≤ 14) (hB : 0 < B) (hH : 0 < H) (hD : 0 < D) (hT : T ≤ 2 ^ 22) (hB13 : B ≤ 8192)
    (hD17 : D ≤ 2 ^ 17) (hBH : B * H < 2 ^ 16) (hvB : vl = true → B = 1)
    (hg : gridX (mk vl B T H D e dt N NT) * (B * H) < 2 ^ 31)
    (hxsz : B * T * H * D * 2 ≤ 2 ^ 36)
    (hosz : (if vl then NT else B * ((T + 2 ^ e - 1) / 2 ^ e)) * H * D * 2 ≤ 2 ^ 38)
    (hdt : dt = .i32 ∨ dt = .i64) (hN : N ≤ 8192) (hNT : 2 * NT * 8 ≤ 2 ^ 39) :
    (mk vl B T H D e dt N NT).ok := by
  have hbytes : dt.bytes ≤ 8 := by rcases hdt with h | h <;> subst h <;> decide
  have h1 : ((B * T * H * D : Nat) : Int) * 2 ≤ 2 ^ 36 := by exact_mod_cast hxsz
  have h2 : (((if vl then NT else B * ((T + 2 ^ e - 1) / 2 ^ e)) * H * D : Nat) : Int) * 2 ≤ 2 ^ 38 := by
    exact_mod_cast hosz
  have h3 : ((N + 1 : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : (N + 1) * dt.bytes ≤ (N + 1) * 8 := Nat.mul_le_mul_left _ hbytes
    have : (N + 1) * 8 ≤ 2 ^ 39 := by omega
    exact_mod_cast le_trans ‹(N + 1) * dt.bytes ≤ (N + 1) * 8› this
  have h4 : ((2 * NT : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : 2 * NT * dt.bytes ≤ 2 * NT * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this hNT
  have hpe : 2 ^ e ≤ 2 ^ 14 := Nat.pow_le_pow_right (by norm_num) he
  refine ⟨⟨e, rfl⟩, ⟨6, rfl⟩, by simp [mk], by simp [mk], ?_, hB, hH, hD, by simp only [mk]; omega, hT, hB13, hD17, hBH,
    hg, hvB, le_refl _, le_refl _, ?_, fun _ => hdt, fun _ => by simp only [mk]; omega, fun _ => hN,
    fun _ => le_refl _, fun _ => le_refl _, fun _ => ?_, fun _ => ?_⟩
  · show 2 ^ e * 64 ≤ 2 ^ 20
    calc 2 ^ e * 64 ≤ 2 ^ 14 * 64 := Nat.mul_le_mul_right _ hpe
      _ = 2 ^ 20 := by norm_num
  · exact bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast at h1 h2 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast at h2 h3 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast at h2 h4 ⊢; omega))

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk false 1 1 1 1 4 .i32 0 0, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl)
      (by norm_num) (by norm_num), ?_⟩
  have hb2 : (DType.bf16.bytes : Int) = 2 := rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · simp [mk]
  · simp only [mk, hb2]; norm_num
  · simp only [mk, hb2]; decide
  · intro x _; simp [zeroMem]
  · simp [mk]
  · simp only [mk, hb2]; norm_num
  · simp only [mk, hb2]; decide
  · intro x _; simp [zeroMem]
  · intro i hi
    have hi0 : i = 0 := by simp only [mk] at hi; omega
    subst hi0
    simp only [mk, Mem.read, Mem.Aligned, Buf.ptr, hb2, Nat.cast_zero]
    norm_num
    exact ⟨[0, 0], rfl, rfl⟩
  · intro hc; simp [mk] at hc

/-! ## Coverage of the host's launches

Every launch of fla's test of the entry point (`tests/ops/utils/test_pooling.py`, `SETUP.md`): the
five fixed-length `(B, T, H, chunk_size, D)` and five variable-length `(T, H, chunk_size, D,
cu_seqlens)` cases, in `bfloat16`, the variable-length ones with `int32` tables (the test's) and
`int64` tables; the pooling of the keys and values in fla's NSA tests (`tests/ops/test_nsa.py`,
`parallel_nsa` calls `mean_pooling(k, block_size, cu_seqlens)` and the same for `v`), in
`bfloat16`; and one packed batch of NSA's default `block_size = 64`. `NT` is the number of chunks
`prepare_chunk_indices` gives. The contents (and so `TablesOk`) are not part of `Inst.ok`; the
host's tables are checked against `TablesOk` on the GPU (`check_gpu.py`). Names:
`setup_<test|nsa>_<fixed|varlen>_<shape>[_<i32|i64>]`. -/

def setup_test_fixed_1x63x1x30_bt16 : Inst := mk false 1 63 1 30 4 .i32 0 0
theorem setup_test_fixed_1x63x1x30_bt16_ok : setup_test_fixed_1x63x1x30_bt16.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_fixed_2x500x4x60_bt32 : Inst := mk false 2 500 4 60 5 .i32 0 0
theorem setup_test_fixed_2x500x4x60_bt32_ok : setup_test_fixed_2x500x4x60_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_fixed_2x1000x5x128_bt64 : Inst := mk false 2 1000 5 128 6 .i32 0 0
theorem setup_test_fixed_2x1000x5x128_bt64_ok : setup_test_fixed_2x1000x5x128_bt64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_fixed_3x1024x6x500_bt64 : Inst := mk false 3 1024 6 500 6 .i32 0 0
theorem setup_test_fixed_3x1024x6x500_bt64_ok : setup_test_fixed_3x1024x6x500_bt64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_fixed_4x2048x8x1024_bt128 : Inst := mk false 4 2048 8 1024 7 .i32 0 0
theorem setup_test_fixed_4x2048x8x1024_bt128_ok : setup_test_fixed_4x2048x8x1024_bt128.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_15x2x60_bt32_n1_i32 : Inst := mk true 1 15 2 60 5 .i32 1 1
theorem setup_test_varlen_15x2x60_bt32_n1_i32_ok : setup_test_varlen_15x2x60_bt32_n1_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_15x2x60_bt32_n1_i64 : Inst := mk true 1 15 2 60 5 .i64 1 1
theorem setup_test_varlen_15x2x60_bt32_n1_i64_ok : setup_test_varlen_15x2x60_bt32_n1_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_test_varlen_1000x3x100_bt64_n3_i32 : Inst := mk true 1 1000 3 100 6 .i32 3 16
theorem setup_test_varlen_1000x3x100_bt64_n3_i32_ok : setup_test_varlen_1000x3x100_bt64_n3_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_1000x3x100_bt64_n3_i64 : Inst := mk true 1 1000 3 100 6 .i64 3 16
theorem setup_test_varlen_1000x3x100_bt64_n3_i64_ok : setup_test_varlen_1000x3x100_bt64_n3_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2000x4x256_bt64_n5_i32 : Inst := mk true 1 2000 4 256 6 .i32 5 35
theorem setup_test_varlen_2000x4x256_bt64_n5_i32_ok : setup_test_varlen_2000x4x256_bt64_n5_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2000x4x256_bt64_n5_i64 : Inst := mk true 1 2000 4 256 6 .i64 5 35
theorem setup_test_varlen_2000x4x256_bt64_n5_i64_ok : setup_test_varlen_2000x4x256_bt64_n5_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2048x4x500_bt128_n5_i32 : Inst := mk true 1 2048 4 500 7 .i32 5 19
theorem setup_test_varlen_2048x4x500_bt128_n5_i32_ok : setup_test_varlen_2048x4x500_bt128_n5_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2048x4x500_bt128_n5_i64 : Inst := mk true 1 2048 4 500 7 .i64 5 19
theorem setup_test_varlen_2048x4x500_bt128_n5_i64_ok : setup_test_varlen_2048x4x500_bt128_n5_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2048x2x1024_bt128_n4_i32 : Inst := mk true 1 2048 2 1024 7 .i32 4 18
theorem setup_test_varlen_2048x2x1024_bt128_n4_i32_ok : setup_test_varlen_2048x2x1024_bt128_n4_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_test_varlen_2048x2x1024_bt128_n4_i64 : Inst := mk true 1 2048 2 1024 7 .i64 4 18
theorem setup_test_varlen_2048x2x1024_bt128_n4_i64_ok : setup_test_varlen_2048x2x1024_bt128_n4_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_nsa_fixed_1x63x1x64_bt32 : Inst := mk false 1 63 1 64 5 .i32 0 0
theorem setup_nsa_fixed_1x63x1x64_bt32_ok : setup_nsa_fixed_1x63x1x64_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_fixed_3x111x1x100_bt32 : Inst := mk false 3 111 1 100 5 .i32 0 0
theorem setup_nsa_fixed_3x111x1x100_bt32_ok : setup_nsa_fixed_3x111x1x100_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_fixed_3x1024x2x60_bt32 : Inst := mk false 3 1024 2 60 5 .i32 0 0
theorem setup_nsa_fixed_3x1024x2x60_bt32_ok : setup_nsa_fixed_3x1024x2x60_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_fixed_3x1024x2x128_bt32 : Inst := mk false 3 1024 2 128 5 .i32 0 0
theorem setup_nsa_fixed_3x1024x2x128_bt32_ok : setup_nsa_fixed_3x1024x2x128_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_fixed_4x2048x2x64_bt32 : Inst := mk false 4 2048 2 64 5 .i32 0 0
theorem setup_nsa_fixed_4x2048x2x64_bt32_ok : setup_nsa_fixed_4x2048x2x64_bt32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_15x1x64_bt32_n1_i32 : Inst := mk true 1 15 1 64 5 .i32 1 1
theorem setup_nsa_varlen_15x1x64_bt32_n1_i32_ok : setup_nsa_varlen_15x1x64_bt32_n1_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_15x1x64_bt32_n1_i64 : Inst := mk true 1 15 1 64 5 .i64 1 1
theorem setup_nsa_varlen_15x1x64_bt32_n1_i64_ok : setup_nsa_varlen_15x1x64_bt32_n1_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_800x1x64_bt16_n4_i32 : Inst := mk true 1 800 1 64 4 .i32 4 51
theorem setup_nsa_varlen_800x1x64_bt16_n4_i32_ok : setup_nsa_varlen_800x1x64_bt16_n4_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_800x1x64_bt16_n4_i64 : Inst := mk true 1 800 1 64 4 .i64 4 51
theorem setup_nsa_varlen_800x1x64_bt16_n4_i64_ok : setup_nsa_varlen_800x1x64_bt16_n4_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_1000x2x64_bt32_n3_i32 : Inst := mk true 1 1000 2 64 5 .i32 3 32
theorem setup_nsa_varlen_1000x2x64_bt32_n3_i32_ok : setup_nsa_varlen_1000x2x64_bt32_n3_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_1000x2x64_bt32_n3_i64 : Inst := mk true 1 1000 2 64 5 .i64 3 32
theorem setup_nsa_varlen_1000x2x64_bt32_n3_i64_ok : setup_nsa_varlen_1000x2x64_bt32_n3_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_2000x2x100_bt32_n5_i32 : Inst := mk true 1 2000 2 100 5 .i32 5 65
theorem setup_nsa_varlen_2000x2x100_bt32_n5_i32_ok : setup_nsa_varlen_2000x2x100_bt32_n5_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_2000x2x100_bt32_n5_i64 : Inst := mk true 1 2000 2 100 5 .i64 5 65
theorem setup_nsa_varlen_2000x2x100_bt32_n5_i64_ok : setup_nsa_varlen_2000x2x100_bt32_n5_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_65536x4x128_bt64_n6_i32 : Inst := mk true 1 65536 4 128 6 .i32 6 1027
theorem setup_nsa_varlen_65536x4x128_bt64_n6_i32_ok : setup_nsa_varlen_65536x4x128_bt64_n6_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inl rfl) (by norm_num) (by norm_num)
def setup_nsa_varlen_65536x4x128_bt64_n6_i64 : Inst := mk true 1 65536 4 128 6 .i64 6 1027
theorem setup_nsa_varlen_65536x4x128_bt64_n6_i64_ok : setup_nsa_varlen_65536x4x128_bt64_n6_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (by decide) (Or.inr rfl) (by norm_num) (by norm_num)

end Claims.MeanPoolingFwd
