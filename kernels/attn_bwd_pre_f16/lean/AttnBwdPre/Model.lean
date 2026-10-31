import TVA

/-!
# `attn_bwd_pre_f16`: the model, the conditions, the specification and the assumptions

Source: `source/06-fused-attention.py`, tutorial 06 (fused attention). The kernel is
`_attn_bwd_preprocess` (lines 248–262); it reads the module-level names `triton` (its decorator)
and `tl`, bound by the imports of lines 20–21. The host's `_attention.backward` (lines 583–623)
launches it (lines 600–607) on the grid `(N_CTX // PRE_BLOCK, BATCH * N_HEAD)` with
`BLOCK_M = PRE_BLOCK = 128` and `HEAD_DIM = ctx.HEAD_DIM`, on the forward output `o` and the
incoming gradient `do` (both `float16`, shape `(BATCH, N_HEAD, N_CTX, HEAD_DIM)`, contiguous) and
a fresh `float32` buffer `delta` of shape `(BATCH, N_HEAD, N_CTX)`. The script runs it from its test
(`test_op`, lines 631–695: `Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`,
`HEAD_DIM ∈ {64, 128}`) and its benchmark (lines 705–775: `BATCH = 4`, `H = 32`,
`N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`); the forward pass admits
`HEAD_DIM ∈ {16, 32, 64, 128, 256}` (line 512).

The kernel computes, for every row `r` of `o` (one `(batch, head, position)`), the `float32` sum
`Delta[r] = Σ_k o[r, k] · do[r, k]` over the `HEAD_DIM` columns.

* The model: `_attn_bwd_preprocess`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with `OutSpec`, the exact sums `dotSum`, `dotAbs` and the
  tolerance `tol`.
* The assumptions: `Assumptions` (one `[rule]` field, `Compiler.Faithful`).

The proof is in `AttnBwdPre.Proof`.
-/

namespace Kernels.AttnBwdPre

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `_attn_bwd_preprocess` (source lines 248–262). `O` and `DO` are `*fp16` and `Delta` is `*fp32`
(the host passes the `float16` tensors `o`, `do` and the `float32` tensor `delta`). `Z`, `H` and
`N_CTX` are the unannotated integer parameters; the body never reads `Z` or `H`, so their class (the
compile-time constant `1`, an `int32`, an `int64` or a `uint64`) changes nothing, and they are
written as `int32` here. `N_CTX` is a runtime `int32` (an integer argument in `[-2^31, 2^31)` other
than `1`; at `1` it is the compile-time constant `1`, which gives the same model, see `model`). The
constexprs `BLOCK_M` and `HEAD_DIM` are Lean parameters. The source's name `do` is a keyword in
Lean, written `«do»`. -/
def _attn_bwd_preprocess (O DO : Ptr .f16) (Delta : Ptr .f32) (Z H N_CTX : I32)
    (BLOCK_M HEAD_DIM : Nat) : Kernel where
  body ctx := do
    -- line 254: axis 0 of the grid, an `int32`, times the constexpr `BLOCK_M` (built in `int32`,
    -- the tensor's dtype), plus `tl.arange(0, BLOCK_M)`: the `int32` block, wrapping arithmetic
    let off_m : Tensor [BLOCK_M] I32 := tl.program_id ctx 0 * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- line 255: axis 1 of the grid, an `int32`
    let off_hz := tl.program_id ctx 1
    -- line 256
    let off_n : Tensor [HEAD_DIM] I32 := tl.arange 0 HEAD_DIM
    -- line 258: `O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]`, left
    -- to right: the `int32` scalar `(off_hz * HEAD_DIM) * N_CTX` (the constexpr built in `int32`;
    -- wrapping products) added to the `*fp16` pointer; then the `int32` block
    -- `off_m[:, None] * HEAD_DIM` (each lane's offset times the constexpr, built in `int32`), whose
    -- broadcast against `off_n[None, :]` (`[BLOCK_M, 1]` with `[1, HEAD_DIM]`) the model writes on
    -- the operand (`.col HEAD_DIM`: lane `(i, k)` holds `off_m[i]`, and `*` acts lane by lane, so
    -- broadcasting first gives the same values); then `off_n[None, :]`, broadcast to every row
    -- (`.row BLOCK_M`). Each addition to a pointer advances it by that many elements. No mask and no
    -- `other`: every lane is enabled.
    let o ← tl.load (O + off_hz * (IntV.ofNat HEAD_DIM : I32) * N_CTX + off_m.col HEAD_DIM * (IntV.ofNat HEAD_DIM : I32) + off_n.row BLOCK_M)
    -- line 259: the same pointers into `DO`, the load, then `.to(tl.float32)`: `float16` to
    -- `float32` with the default rounding (`tl.cvt`, a widening, exact)
    let «do» : Tensor [BLOCK_M, HEAD_DIM] F32 ← tl.cvt (← tl.load (DO + off_hz * (IntV.ofNat HEAD_DIM : I32) * N_CTX + off_m.col HEAD_DIM * (IntV.ofNat HEAD_DIM : I32) + off_n.row BLOCK_M))
    -- line 260: `o * do` multiplies a `float16` block by a `float32` block, so it computes in
    -- `float32` (`BinOp.typing? .mul`): `o` is converted first (`tl.convert .f16 .f32`), then the
    -- IEEE `float32` product (`tl.mul`). `tl.sum(…, axis=1)` sums a product, so its additions may be
    -- contracted with it (`enable_fp_fusion`): `tl.sum1c`, along a tree Triton chooses, in
    -- `float32` (a float sum does not widen)
    let delta ← tl.sum1c (← tl.mul (← tl.convert .f16 .f32 o) «do»)
    -- line 262: `Delta + off_hz * N_CTX + off_m`: the `int32` scalar `off_hz * N_CTX` added to the
    -- `*fp32` pointer, then the `int32` block `off_m`; `delta` is already `float32`, the pointer's
    -- dtype, so no conversion; no mask
    tl.store (Delta + off_hz * N_CTX + off_m) delta

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `_attn_bwd_preprocess`. -/
structure Inst where
  /-- `Z` (the host passes `BATCH = q.shape[0]`). -/
  Z : Nat
  /-- `H` (the host passes `N_HEAD = q.shape[1]`). -/
  H : Nat
  /-- `N_CTX` (the host passes `q.shape[2]`). -/
  N_CTX : Nat
  /-- `HEAD_DIM` (the host passes `ctx.HEAD_DIM`, `q.shape[-1]`). -/
  HEAD_DIM : Nat
  /-- `BLOCK_M` (the host passes `PRE_BLOCK = 128`). -/
  BLOCK_M : Nat
  /-- `O`: the forward output `o`, `float16`; element `(b, h, m, k)` of the `(Z, H, N_CTX,
  HEAD_DIM)` tensor is element `((b·H + h)·N_CTX + m)·HEAD_DIM + k` of the buffer. -/
  O : Buf .f16
  /-- `DO`: the gradient `do`, `float16`, laid out as `O`. -/
  DO : Buf .f16
  /-- `Delta`: the output `delta`, `float32`; element `(b, h, m)` of the `(Z, H, N_CTX)` tensor is
  element `(b·H + h)·N_CTX + m` of the buffer. -/
  Delta : Buf .f32

/-- The rows the launch computes: the elements `r < Z·H·N_CTX` of `Delta`. -/
def outSet (d : Inst) : Set Nat := {r | r < d.Z * d.H * d.N_CTX}

/-- The input elements the launch reads: the elements `e < Z·H·N_CTX·HEAD_DIM` of `O` and `DO`
(row `r`, column `k` is element `r·HEAD_DIM + k`). -/
def inSet (d : Inst) : Set Nat := {e | e < d.Z * d.H * d.N_CTX * d.HEAD_DIM}

/-- The instances covered. The shapes are variables; the bounds are those of Triton's block shapes
and CUDA's grid, the kernel's own `int32` index arithmetic, and the tolerance's meaning
(`HEAD_DIM ≤ 256`). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK_M` is a power of two (Triton requires it of `tl.arange`; the host's `128`). -/
  block_pow2 : ∃ e, d.BLOCK_M = 2 ^ e
  /-- `HEAD_DIM` is a power of two (Triton requires it of `tl.arange`; the host's `16 … 256`). -/
  head_pow2 : ∃ e, d.HEAD_DIM = 2 ^ e
  /-- `HEAD_DIM ≤ 256`: the largest head dimension the host admits (line 512); the tolerance of
  `Post` grows with `HEAD_DIM` and is stated up to it (`tol_le`). -/
  head_le : d.HEAD_DIM ≤ 256
  /-- The block `[BLOCK_M, HEAD_DIM]` has at most `2^20` elements (Triton's largest block). -/
  block_le : d.BLOCK_M * d.HEAD_DIM ≤ 2 ^ 20
  /-- `BLOCK_M` divides `N_CTX` (the host asserts `N_CTX % PRE_BLOCK == 0`, line 599), so the grid
  `N_CTX // BLOCK_M` covers every position exactly. -/
  ctx_dvd : d.BLOCK_M ∣ d.N_CTX
  /-- `N_CTX < 2^31`: Triton passes it as an `int32`. -/
  ctx_lt : d.N_CTX < 2 ^ 31
  /-- `Z·H < 2^16`: CUDA's limit on the grid's axis 1, `BATCH * N_HEAD` (`Launch.Valid`). -/
  bh_lt : d.Z * d.H < 2 ^ 16
  /-- The grid has fewer than `2^31` programs (`Launch.Valid`). -/
  grid_lt : d.N_CTX / d.BLOCK_M * (d.Z * d.H) < 2 ^ 31
  /-- The `int32` products `off_m * HEAD_DIM` (lines 258–259) do not wrap: `off_m < N_CTX`. -/
  row_fit : (d.N_CTX - 1) * d.HEAD_DIM < 2 ^ 31
  /-- The `int32` products `off_hz * HEAD_DIM * N_CTX` (lines 258–259) do not wrap:
  `off_hz < Z·H`. (They bound `off_hz * N_CTX`, line 262, too.) -/
  head_fit : (d.Z * d.H - 1) * d.HEAD_DIM * d.N_CTX < 2 ^ 31
  /-- `O` holds every element the launch reads. -/
  O_len : d.Z * d.H * d.N_CTX * d.HEAD_DIM ≤ d.O.len
  /-- `DO` holds every element the launch reads. -/
  DO_len : d.Z * d.H * d.N_CTX * d.HEAD_DIM ≤ d.DO.len
  /-- `Delta` holds every element the launch writes. -/
  Delta_len : d.Z * d.H * d.N_CTX ≤ d.Delta.len
  /-- The elements of `Delta` the launch writes share no byte with the elements of `O` it reads. -/
  Delta_O : Disjoint (d.Delta.elems (outSet d)) (d.O.elems (inSet d))
  /-- The elements of `Delta` the launch writes share no byte with the elements of `DO` it reads. -/
  Delta_DO : Disjoint (d.Delta.elems (outSet d)) (d.DO.elems (inSet d))

/-- The model on an instance: the buffers' first elements, the three integers as `int32`, and the
two constexprs. `Z` and `H` are never read. `N_CTX` is `1` only when `BLOCK_M = 1` (`ctx_dvd`), and
then Triton passes the compile-time constant `1`; its two uses, `(off_hz * HEAD_DIM) * N_CTX` and
`off_hz * N_CTX`, multiply an `int32` scalar by it, which builds it in `int32` (SEMANTICS.md,
"Constants": a Python number with an arithmetic operator takes the operation's dtype), the value
the runtime `int32` `1` has, so the model is the same. -/
def model (d : Inst) : Kernel :=
  _attn_bwd_preprocess (d.O.ptr 0) (d.DO.ptr 0) (d.Delta.ptr 0) (IntV.ofNat d.Z) (IntV.ofNat d.H)
    (IntV.ofNat d.N_CTX) d.BLOCK_M d.HEAD_DIM

/-- The host's launch (lines 600–607): the grid `(N_CTX // PRE_BLOCK, BATCH * N_HEAD)`, a 2-tuple,
so axis 2 has size `1` (Python's `//` on non-negative integers is `Nat` division). The source's
block shapes: `[BLOCK_M]` (`tl.arange(0, BLOCK_M)`, `off_m`, `delta`, the `Delta` pointers),
`[HEAD_DIM]` (`off_n`), `[BLOCK_M, 1]` (`off_m[:, None]`, its product, the pointers before
`off_n` is added), `[1, HEAD_DIM]` (`off_n[None, :]`) and `[BLOCK_M, HEAD_DIM]` (the `O` and `DO`
pointers, `o`, `do`, `o * do`). The launch gives no option, so every one is Triton's default:
`enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, `launch_pdl = False`. -/
def launch (d : Inst) : Launch where
  grid := ![d.N_CTX / d.BLOCK_M, d.Z * d.H, 1]
  fusion := true
  debug := false
  numCtas := 1
  pdl := false
  blocks := [[d.BLOCK_M], [d.HEAD_DIM], [d.BLOCK_M, 1], [1, d.HEAD_DIM], [d.BLOCK_M, d.HEAD_DIM]]

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present) in the initial memory. Any `float16` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.O.Valid init ∧ d.DO.Valid init ∧ d.Delta.Valid init

/-! ## The specification -/

/-- The exact dot product of a row of `o` (`vo k`) with the same row of `do` (`vdo k`), `k < D`. -/
def dotSum (vo vdo : Nat → F16) (D : Nat) : ℚ := ∑ k ∈ Finset.range D, (vo k).toRat * (vdo k).toRat

/-- The sum of the products' magnitudes, `Σ_k |o[k] · do[k]|`. -/
def dotAbs (vo vdo : Nat → F16) (D : Nat) : ℚ := ∑ k ∈ Finset.range D, |(vo k).toRat * (vdo k).toRat|

/-- The tolerance of one output, for `D = HEAD_DIM` products of absolute sum `B`, with
`u = 2^-24` and `η = 2^-150` (one `float32` rounding's relative and absolute bounds):
`errBound(3u, 4η, D − 1, D − 1, (1 + u)·B + D·η) + (u·B + D·η)`, where
`Tree.errBound δ η d N A = ((1 + δ)^d − 1)·A + η·N·(1 + δ)^d`. The second term covers the products'
roundings (`u·|a·b| + η` each); the first, the sum of the rounded products along any tree of `D`
leaves (depth and node count at most `D − 1`) whose additions may be contracted with the products
(`3u` relative and `4η` absolute per addition, `FloatV.AddRel.err`). It is the library's bound
`bcastTreeBound D B` (`tol_eq`), and below `2^-14·B + 2^-138` for `D ≤ 256` (`tol_le`). -/
def tol (D : Nat) (B : ℚ) : ℚ :=
  Tree.errBound (3 * 2 ^ (-24 : ℤ)) (4 * 2 ^ (-150 : ℤ)) (D - 1) (D - 1)
      ((1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ)) +
    (2 ^ (-24 : ℤ) * B + D * 2 ^ (-150 : ℤ))

/-- What the launch leaves in element `r` of `Delta`: some `float32` value `c`; and when row `r` of
`O` and of `DO` (elements `r·HEAD_DIM + k`, `k < HEAD_DIM`) hold finite values `vo k`, `vdo k` in the
initial memory, `c` is finite and within `tol HEAD_DIM (Σ_k |vo k · vdo k|)` of the exact
`Σ_k vo k · vdo k`. -/
def OutSpec (d : Inst) (init fin : Mem) (r : Nat) : Prop :=
  ∃ c : F32, fin.read .f32 (d.Delta.ptr ↑r).addr = some c ∧
    ∀ vo vdo : Nat → F16,
      (∀ k, k < d.HEAD_DIM → init.read .f16 (d.O.ptr ↑(r * d.HEAD_DIM + k)).addr = some (vo k) ∧
        (vo k).isFinite = true) →
      (∀ k, k < d.HEAD_DIM → init.read .f16 (d.DO.ptr ↑(r * d.HEAD_DIM + k)).addr = some (vdo k) ∧
        (vdo k).isFinite = true) →
      c.isFinite = true ∧
        |c.toRat - dotSum vo vdo d.HEAD_DIM| ≤ tol d.HEAD_DIM (dotAbs vo vdo d.HEAD_DIM)

/-- What the launch leaves: for every `(batch, head)` pair `bh < Z·H` and position `m < N_CTX`,
element `bh·N_CTX + m` of `Delta` as `OutSpec` says (row `bh·N_CTX + m` of `O` and `DO`); every
byte outside those elements of `Delta` unchanged (`O`, `DO`, and `Delta` beyond them). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ bh m, bh < d.Z * d.H → m < d.N_CTX → OutSpec d init fin (bh * d.N_CTX + m)) ∧
    ∀ z, z ∉ d.Delta.elems (outSet d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

end Kernels.AttnBwdPre
