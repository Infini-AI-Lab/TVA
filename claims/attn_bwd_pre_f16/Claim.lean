import TVA

/-!
# `attn_bwd_pre_f16`: the prepared claim

Source: `source/06-fused-attention.py` (tutorial 06, fused attention).
The kernel is `_attn_bwd_preprocess` (lines 248–262); it reads the module-level names `triton`
(its decorator) and `tl`, bound by the imports of lines 20–21. The host's `_attention.backward`
(lines 582–623) launches it (lines 600–607) on the grid `(N_CTX // PRE_BLOCK, BATCH * N_HEAD)` with
`BLOCK_M = PRE_BLOCK = 128` and `HEAD_DIM = ctx.HEAD_DIM`, on the forward output `o` and the
incoming gradient `do` (both `float16`, shape `(BATCH, N_HEAD, N_CTX, HEAD_DIM)`, contiguous) and a
fresh `float32` buffer `delta` of shape `(BATCH, N_HEAD, N_CTX)`. The launches to cover are in
`SETUP.md` (the file's test and benchmark).

The kernel computes, for every row `r` of `o` (one `(batch, head, position)`), the `float32` sum
`Delta[r] = Σ_k o[r, k] · do[r, k]` over the `HEAD_DIM` columns.

* The model: `_attn_bwd_preprocess`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with `OutSpec`, the exact sums `dotSum`, `dotAbs` and the
  tolerance `tol` (`tol_le_bcast`: below the library's `bcastTreeBound`; `tol_le`: its size).
* The assumptions: `Assumptions` (`Compiler.Faithful` and the depth of a `tl.sum`'s tree).
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `canon_ok`, `setup_*_ok`.
-/

namespace Claims.AttnBwdPre

open TVA TVA.Math.Float

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
`errBound(3u, 4η, ⌈log₂ D⌉, D − 1, (1 + u)·B + D·η) + (u·B + D·η)`, where
`Tree.errBound δ η d N A = ((1 + δ)^d − 1)·A + η·N·(1 + δ)^d`. The second term covers the products'
roundings (`u·|a·b| + η` each); the first, the sum of the rounded products along a tree of `D`
leaves of depth at most `⌈log₂ D⌉` (`Rules.Num.SumDepth`) and `D − 1` nodes, whose additions may be
contracted with the products (`3u` relative and `4η` absolute per addition, `FloatV.AddRel.err`).
It is `sum_bcast_mul_tree_err`'s bound `bcastTreeBound D B` with the depth `⌈log₂ D⌉` in place of
`D − 1` (`tol_le_bcast`), and below `2^-19·B + 2^-138` for `D ≤ 256` (`tol_le`). -/
def tol (D : Nat) (B : ℚ) : ℚ :=
  Tree.errBound (3 * 2 ^ (-24 : ℤ)) (4 * 2 ^ (-150 : ℤ)) (Nat.clog 2 D) (D - 1)
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
  /-- [rule] `Num.SumDepth`: the tree of the float `tl.sum(o * do, axis=1)` (line 260) over its
  `HEAD_DIM` positions has depth at most `⌈log₂ HEAD_DIM⌉`. The kernel has no loop. -/
  sum_depth : Rules.Num.SumDepth P

/-! ## The tolerance -/

/-- `tol` is at most the library's bound for rounded products summed along any tree with fusable
additions (`bcastTreeBound`, depth `D − 1`): the depth `⌈log₂ D⌉` is smaller. -/
theorem tol_le_bcast (D : Nat) {B : ℚ} (hB : 0 ≤ B) : tol D B ≤ bcastTreeBound D B := by
  unfold tol bcastTreeBound
  rw [Format.binary32_u, Format.binary32_eta]
  have hδ : (0 : ℚ) ≤ 3 * 2 ^ (-24 : ℤ) := by positivity
  have hη : (0 : ℚ) ≤ 4 * 2 ^ (-150 : ℤ) := by positivity
  have hA : 0 ≤ (1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ) := by positivity
  have hd : Nat.clog 2 D ≤ D - 1 := by
    rcases Nat.lt_or_ge D 1 with h | h
    · rw [Nat.clog_of_right_le_one (by omega)]; omega
    · refine Nat.clog_le_of_le_pow ?_
      have := Nat.lt_two_pow_self (n := D - 1)
      omega
  exact add_le_add (Tree.errBound_mono hδ hη hd (le_refl (D - 1)) hA le_rfl) le_rfl

/-- The tolerance grows with `D`. -/
theorem tol_mono {D D' : Nat} (hD : D ≤ D') {B : ℚ} (hB : 0 ≤ B) : tol D B ≤ tol D' B := by
  unfold tol
  have hδ : (0 : ℚ) ≤ 3 * 2 ^ (-24 : ℤ) := by positivity
  have hη : (0 : ℚ) ≤ 4 * 2 ^ (-150 : ℤ) := by positivity
  have hDD : (D : ℚ) ≤ D' := by exact_mod_cast hD
  have hA : 0 ≤ (1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ) := by positivity
  have hAA : (1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ) ≤ (1 + 2 ^ (-24 : ℤ)) * B + D' * 2 ^ (-150 : ℤ) := by
    have : (D : ℚ) * 2 ^ (-150 : ℤ) ≤ D' * 2 ^ (-150 : ℤ) := mul_le_mul_of_nonneg_right hDD (by positivity)
    linarith
  have h1 := Tree.errBound_mono hδ hη (d := Nat.clog 2 D) (d' := Nat.clog 2 D') (N := D - 1)
    (N' := D' - 1) (Nat.clog_mono_right 2 hD) (by omega) hA hAA
  have h2 : (D : ℚ) * 2 ^ (-150 : ℤ) ≤ D' * 2 ^ (-150 : ℤ) := mul_le_mul_of_nonneg_right hDD (by positivity)
  linarith

/-- **The tolerance at the covered sizes**: for `HEAD_DIM ≤ 256`, `tol` is below
`2^-19·B + 2^-138`, where `B = Σ_k |o[k]·do[k]|`. -/
theorem tol_le {D : Nat} (hD : D ≤ 256) {B : ℚ} (hB : 0 ≤ B) :
    tol D B ≤ 2 ^ (-19 : ℤ) * B + 2 ^ (-138 : ℤ) := by
  refine (tol_mono hD hB).trans ?_
  have hc : Nat.clog 2 256 = 8 := by
    rw [show (256 : ℕ) = 2 ^ 8 by norm_num]; exact Nat.clog_pow 2 8 (by norm_num)
  unfold tol Tree.errBound
  rw [hc]
  norm_num
  linarith


/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.sumDepth⟩⟩

/-- Elements of two buffers whose byte ranges do not meet do not meet. -/
theorem elems_disjoint_of_lt {d₁ d₂ : DType} {b : Buf d₁} {c : Buf d₂} {S T : Set Nat}
    (hS : ∀ j ∈ S, j < b.len) (hT : ∀ j ∈ T, j < c.len)
    (h : b.base + b.len * d₁.bytes ≤ c.base ∨ c.base + c.len * d₂.bytes ≤ b.base) :
    Disjoint (b.elems S) (c.elems T) := by
  refine Buf.elems_disjoint_of b c ?_ hS hT
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-! ## The host's layout

`canon Z H N D` is the host's launch (lines 600–607) for `BATCH = Z`, `N_HEAD = H`, `N_CTX = N`,
`HEAD_DIM = D`: `BLOCK_M = PRE_BLOCK = 128`, contiguous `o` and `do` of `Z·H·N·D` elements and a
fresh `delta` of `Z·H·N`, the three storages at `2^40`, `2·2^40` and `3·2^40`. -/

def canon (Z H N D : Nat) : Inst where
  Z := Z
  H := H
  N_CTX := N
  HEAD_DIM := D
  BLOCK_M := 128
  O := ⟨2 ^ 40, Z * H * N * D⟩
  DO := ⟨2 * 2 ^ 40, Z * H * N * D⟩
  Delta := ⟨3 * 2 ^ 40, Z * H * N⟩

/-- **The host's launches are covered**: any `HEAD_DIM` the forward admits (`16 … 256`, line
512), any `N_CTX` with `N_CTX % 128 == 0` (line 599), and any `BATCH`, `N_HEAD` whose sizes keep
the grid within CUDA's limits and the kernel's `int32` offsets from wrapping (the numeric
hypotheses, which `norm_num` checks on each launch). -/
theorem canon_ok (Z H N D a : Nat) (hD : D = 2 ^ a) (ha : a ≤ 8) (hN : 128 ∣ N)
    (hN31 : N < 2 ^ 31) (hbh : Z * H < 2 ^ 16) (hgrid : N / 128 * (Z * H) < 2 ^ 31)
    (hrow : (N - 1) * D < 2 ^ 31) (hhead : (Z * H - 1) * D * N < 2 ^ 31)
    (hsize : Z * H * N * D < 2 ^ 39) : (canon Z H N D).ok := by
  have hD256 : D ≤ 256 := by
    rw [hD]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
      _ = 256 := by norm_num
  have hZHN : Z * H * N ≤ Z * H * N * D := by
    rcases Nat.eq_zero_or_pos D with h0 | h0
    · rw [hD] at h0; exact absurd h0 (by positivity)
    · exact Nat.le_mul_of_pos_right _ h0
  refine
    { block_pow2 := ⟨7, rfl⟩
      head_pow2 := ⟨a, hD⟩
      head_le := hD256
      block_le := by simp only [canon]; omega
      ctx_dvd := hN
      ctx_lt := hN31
      bh_lt := hbh
      grid_lt := hgrid
      row_fit := hrow
      head_fit := hhead
      O_len := le_rfl
      DO_len := le_rfl
      Delta_len := le_rfl
      Delta_O := ?_
      Delta_DO := ?_ }
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, canon] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, canon] at hj ⊢; omega)
      (by simp only [canon, DType.bytes]; omega)
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, canon] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, canon] at hj ⊢; omega)
      (by simp only [canon, DType.bytes]; omega)

/-! ## Coverage: the launches of `SETUP.md`

The test (`test_op`, lines 631–695, `mode = "bwd"`, `provider = "triton-fp16"`): `Z ∈ {1, 4}`,
`H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, both `causal` (which this
kernel does not read). The benchmark (`bench_flash_attention`, lines 705–775, `mode = "bwd"`,
`provider = "triton-fp16"`): `BATCH = 4`, `N_HEADS = 32`, `N_CTX = 2^10 … 2^14`,
`HEAD_DIM ∈ {64, 128}`, both `causal`. -/

/-- One launch of the test. -/
def setup_test (Z H N D : Nat) : Inst := canon Z H N D

theorem setup_test_ok : ∀ Z ∈ [1, 4], ∀ H ∈ [2, 48], ∀ N ∈ [128, 1024, 4096], ∀ D ∈ [64, 128],
    (setup_test Z H N D).ok := by
  intro Z hZ H hH N hN D hD
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hZ hH hN hD
  rcases hD with rfl | rfl
  · rcases hZ with rfl | rfl <;> rcases hH with rfl | rfl <;> rcases hN with rfl | rfl | rfl <;>
      exact canon_ok _ _ _ _ 6 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rcases hZ with rfl | rfl <;> rcases hH with rfl | rfl <;> rcases hN with rfl | rfl | rfl <;>
      exact canon_ok _ _ _ _ 7 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- One launch of the benchmark: `BATCH = 4`, `N_HEADS = 32`, `N_CTX = 2^e`. -/
def setup_bench (e D : Nat) : Inst := canon 4 32 (2 ^ e) D

theorem setup_bench_ok : ∀ e ∈ [10, 11, 12, 13, 14], ∀ D ∈ [64, 128], (setup_bench e D).ok := by
  intro e he D hD
  simp only [List.mem_cons, List.not_mem_nil, or_false] at he hD
  rcases hD with rfl | rfl
  · rcases he with rfl | rfl | rfl | rfl | rfl <;>
      exact canon_ok _ _ _ _ 6 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rcases he with rfl | rfl | rfl | rfl | rfl <;>
      exact canon_ok _ _ _ _ 7 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The test's largest launch: `Z = 4`, `H = 48`, `N_CTX = 4096`, `HEAD_DIM = 128`. -/
def setup_test_large : Inst := setup_test 4 48 4096 128
theorem setup_test_large_ok : setup_test_large.ok :=
  setup_test_ok 4 (by simp) 48 (by simp) 4096 (by simp) 128 (by simp)

/-- The benchmark's largest launch: `BATCH = 4`, `N_HEADS = 32`, `N_CTX = 2^14`, `HEAD_DIM = 128`. -/
def setup_bench_large : Inst := setup_bench 14 128
theorem setup_bench_large_ok : setup_bench_large.ok :=
  setup_bench_ok 14 (by simp) 128 (by simp)

/-- The other head dims the forward admits (line 512): `16`, `32`, `256`, at the test's largest
heads and the benchmark's longest context. -/
theorem setup_headdims_ok : ∀ a ∈ [4, 5, 8], (canon 4 48 4096 (2 ^ a)).ok ∧ (canon 4 32 16384 (2 ^ a)).ok := by
  intro a ha
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
  rcases ha with rfl | rfl | rfl <;>
    exact ⟨canon_ok _ _ _ _ _ rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num),
      canon_ok _ _ _ _ _ rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num)⟩

/-! ## Satisfiable -/

/-- One instance of the source's test: `Z = 1`, `H = 2`, `N_CTX = 128`, `HEAD_DIM = 64`. -/
def testMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_test 1 2 128 64, testMem, setup_test_ok 1 (by simp) 2 (by simp) 128 (by simp) 64 (by simp),
    ?_, ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [setup_test, canon], by norm_num [setup_test, canon, DType.bytes],
      by norm_num [setup_test, canon, DType.bytes], fun _ _ => by simp [testMem]⟩

end Claims.AttnBwdPre
