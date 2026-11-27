import TVA

/-!
# `l2norm_bwd_bf16`: L2 normalisation of rows, backward (fla)

The kernel is `l2norm_bwd_kernel` of `source/l2norm.py` (lines 51–67; it reads the module-level
names `triton`, its decorator, and `tl`, bound by the imports of lines 47–48). Its host
`l2norm_bwd` (lines 70–95) launches it on the grid `(triton.cdiv(T, 16),)` with `BT = 16` and
`num_warps = 4`, on the forward's output `y` viewed as `[T, D]` (`bfloat16`, or `float32` when the
forward ran with `output_dtype = torch.float32`), its `float32` factors `rstd` (`T` elements), the
incoming gradient `dy` laid out as `y` (`bfloat16` or `float32`), and a fresh `dx` laid out as `y`,
of `y`'s dtype; `SETUP.md` lists the launches. Each row `t < T` gets
`dx = dy·rstd − (Σ_j dy_j·y_j)·y·rstd`, computed in `float32` over the padded row length `BD`.

What is claimed: every covered launch ends normally, and for each row `t < T`, with `a_j`, `g_j`
the row's initial `y` and `dy` values, `r` its initial `rstd`, `S = Σ_j g_j·a_j`,
`A = Σ_j |g_j·a_j|`, the exact `ref_j = g_j·r − S·a_j·r`, the scale `M_j = |r|·(|g_j| + A·|a_j|)`
and `F_j = (1 + |r|)·(1 + |a_j|)`: each `dx[t, j]`, `j < D`, is a finite value of `dx`'s dtype
with `|dx[t, j] − ref_j| ≤ 2^-18·M_j + 2^-138·F_j` for a `float32` `dx`, and
`≤ 2^-8·|ref_j| + 2^-17·M_j + 2^-133·F_j` for a `bfloat16` `dx`; every byte outside the elements
of `dx` the launch writes is unchanged.

The conditions: `1 ≤ D ≤ 512` with `BD` a power of two, `D ≤ BD ≤ 512`; `BT` a power of two with
`BT·BD ≤ 2^20`; fewer than `2^31` programs; the buffers hold the `T·D` (`y`, `dy`, `dx`) and `T`
(`rstd`) elements the launch touches, and the written elements of `dx` share no byte with the
read elements of `y`, `dy` and `rstd`; every element read is finite, and on every row
`(1 + |r|)·(1 + |a_j|)·(1 + |g_j| + A) ≤ 2^124` for each `j < D`.

The tolerances rest on no approximate device operation: the kernel's float operations are IEEE
products, fusable additions and roundings; the depth of the row sum comes from the device fact
`Device.SumDepth`.
-/

namespace Claims.L2NormBwd

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `l2norm_bwd_kernel` (source lines 51–67). `rstd` is `*fp32`; `y`, `dy` and `dx` are each
`*bf16` (`Sum.inl`) or `*fp32` (`Sum.inr`): the host passes the forward's `y` (`bfloat16`, or
`float32` for `output_dtype = torch.float32`), its gradient `dy`, and `dx = empty_like(y)`; the
pointers' dtypes are part of Triton's specialisation, so each combination is its own compiled
kernel, and the model takes them as the cases of `y`, `dy`, `dx`. `eps` is a Python `float`
argument (a runtime `float32` scalar, `FloatV.hostF32` of the host's double) that the body never
reads. `T` is an unannotated integer argument listed in `do_not_specialize`, so it is never the
compile-time constant `1`: a runtime `int32` for `T < 2^31` and a runtime `int64` from `2^31`; the
model takes it as either (`Sum.inl`, `Sum.inr`), as `model` picks by its value. `D`, `BD`, `NB`
and `BT` are `tl.constexpr`s, Lean parameters; the body never reads `NB` (an autotune key only). -/
def l2norm_bwd_kernel (y : Sum (Ptr .bf16) (Ptr .f32)) (rstd : Ptr .f32)
    (dy : Sum (Ptr .bf16) (Ptr .f32)) (dx : Sum (Ptr .bf16) (Ptr .f32))
    (eps : F32) (T : Sum I32 I64) (D BD NB BT : Nat) : Kernel where
  body ctx := do
    -- line 53: `i_t = tl.program_id(0).to(tl.int64)`: axis 0 of the grid, an `int32` scalar,
    -- sign-extended to `int64`
    let i_t : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 54: `i_t * BT + tl.arange(0, BT)`: the constexpr `BT` built in `int64` (the dtype of
    -- the scalar tensor `i_t`), the `int64` product; the `int32` block `tl.arange(0, BT)`
    -- sign-extended to `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [BT] I64 ← tl.convert .i32 .i64 (tl.arange 0 BT)
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + o_a
    -- line 55: an `int32` block
    let o_d : Tensor [BD] I32 := tl.arange 0 BD
    -- line 56: `m_t = o_t < T`: the scalar `T` promoted to `int64` (a sign extension when it is an
    -- `int32`; nothing when it is already an `int64`), then the signed comparison
    let T64 : I64 ← match T with
      | .inl t => do pure (← tl.convert .i32 .i64 (Tensor.scalar t)).item
      | .inr t => pure t
    let m_t : Tensor [BT] Bool := o_t.ltS T64
    -- line 57: `m_t[:, None] & (o_d[None, :] < D)`: `o_d[None, :]` against the constexpr `D`
    -- built in `int32` (the tensor's dtype), then the `&` of the `[BT, 1]` and `[1, BD]` blocks,
    -- broadcast to `[BT, BD]`; the model broadcasts the operands (`.col`, `.row`) and takes `&`
    -- lane by lane, the same lane values
    let m_d : Tensor [BD] Bool := o_d.ltS (IntV.ofNat D : I32)
    let m_x : Tensor [BT, BD] Bool := (m_t.col BD).and (m_d.row BT)
    -- line 58: `y + o_t[:, None] * D + o_d[None, :]`, evaluated left to right: the `int64` block
    -- `o_t[:, None]` times the constexpr `D` (built in `int64`), added to the pointer, then the
    -- `int32` offsets `o_d[None, :]` added to those pointers (each pointer advanced by its lane's
    -- offset, the `[BT, 1]` and `[1, BD]` operands broadcast to `[BT, BD]` first, the same lane
    -- values); of `y`'s dtype
    let p_y : Sum (Tensor [BT, BD] (Ptr .bf16)) (Tensor [BT, BD] (Ptr .f32)) := match y with
      | .inl y => .inl (y + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
      | .inr y => .inr (y + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
    -- line 59: the `*fp32` pointer advanced by the `int64` block `o_t`
    let p_rstd : Tensor [BT] (Ptr .f32) := rstd + o_t
    -- lines 60–61: the same pointers into `dy` and `dx`, of their dtypes
    let p_dy : Sum (Tensor [BT, BD] (Ptr .bf16)) (Tensor [BT, BD] (Ptr .f32)) := match dy with
      | .inl dy => .inl (dy + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
      | .inr dy => .inr (dy + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
    let p_dx : Sum (Tensor [BT, BD] (Ptr .bf16)) (Tensor [BT, BD] (Ptr .f32)) := match dx with
      | .inl dx => .inl (dx + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
      | .inr dx => .inr (dx + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
    -- line 63: the masked load with `other=0.0`: the Python `0.0` is a `float32` tensor (`+0`),
    -- converted to the pointer's dtype (`+0`, exact: the value `litBF16`, `litF32` of `0` gives);
    -- then `.to(tl.float32)`: from `bfloat16` the exact widening (`tl.cvt`), from `float32` the
    -- tensor itself
    let b_y : Tensor [BT, BD] F32 ← match p_y with
      | .inl p_y => do
        let b_y16 : Tensor [BT, BD] BF16 ←
          tl.load p_y (some m_x) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
        tl.cvt b_y16
      | .inr p_y => tl.load p_y (some m_x) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- line 64: the masked `float32` load with `other=0.0` (`+0`); `.to(tl.float32)` is the
    -- tensor itself
    let b_rstd : Tensor [BT] F32 ←
      tl.load p_rstd (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- line 65: as line 63, for `dy`
    let b_dy : Tensor [BT, BD] F32 ← match p_dy with
      | .inl p_dy => do
        let b_dy16 : Tensor [BT, BD] BF16 ←
          tl.load p_dy (some m_x) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
        tl.cvt b_dy16
      | .inr p_dy => tl.load p_dy (some m_x) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- line 66: `b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]`,
    -- evaluated left to right.
    -- `b_dy * b_rstd[:, None]`: `b_rstd` broadcast to the columns, the IEEE `float32` product.
    -- `tl.sum(b_dy * b_y, 1)`: the IEEE products, summed along axis 1; with `enable_fp_fusion`
    -- the additions may be contracted with the products: the fusable `tl.sum1c`, along a tree
    -- Triton chooses.
    -- `…[:, None] * b_y * b_rstd[:, None]`: the row sums broadcast to the columns, times `b_y`,
    -- then times `b_rstd` broadcast (left to right), two IEEE products.
    -- `… - …`: both operands are products, so the backend may contract either into an `fma`:
    -- the fusable `tl.subc`.
    let t1 : Tensor [BT, BD] F32 ← tl.mul b_dy (b_rstd.col BD)
    let pr : Tensor [BT, BD] F32 ← tl.mul b_dy b_y
    let s : Tensor [BT] F32 ← tl.sum1c pr
    let t2 : Tensor [BT, BD] F32 ← tl.mul (s.col BD) b_y
    let t3 : Tensor [BT, BD] F32 ← tl.mul t2 (b_rstd.col BD)
    let b_dx : Tensor [BT, BD] F32 ← tl.subc t1 t3
    -- line 67: `b_dx.to(p_dx.dtype.element_ty)`: to `bfloat16`, one rounding to nearest even
    -- (`tl.cvt`); to `float32`, `b_dx`'s own dtype, `b_dx` itself; then the masked store
    match p_dx with
    | .inl p_dx => do
      let b_dx16 : Tensor [BT, BD] BF16 ← tl.cvt b_dx
      tl.store p_dx b_dx16 (some m_x)
    | .inr p_dx => tl.store p_dx b_dx (some m_x)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `l2norm_bwd_kernel`. -/
structure Inst where
  /-- `T`: the rows of the view `y.view(-1, dy.shape[-1])` (batch × tokens × heads). -/
  T : Nat
  /-- `D`: `dy.shape[-1]`, the row length. -/
  D : Nat
  /-- `BD`: the host's `min(65536 // y.element_size(), triton.next_power_of_2(D))`. -/
  BD : Nat
  /-- `NB`: the host's `triton.cdiv(T, 2048 * 32)`, an autotune key the body never reads. -/
  NB : Nat
  /-- `BT`: the rows per program (the host passes `16`). -/
  BT : Nat
  /-- `eps`: the host's Python `float` (a double); the kernel gets `FloatV.hostF32 eps` and never
  reads it. -/
  eps : F64
  /-- `y`: the forward's output, contiguous, `bfloat16` (`Sum.inl`) or `float32` (`Sum.inr`);
  `y[t, j]` is element `t·D + j`. -/
  y : Sum (Buf .bf16) (Buf .f32)
  /-- `rstd`: the forward's factors, `float32`, element `t` for row `t`. -/
  rstd : Buf .f32
  /-- `dy`: the incoming gradient, laid out as `y`, `bfloat16` or `float32`. -/
  dy : Sum (Buf .bf16) (Buf .f32)
  /-- `dx`: the output, laid out as `y`, `bfloat16` or `float32` (the host's `empty_like(y)`). -/
  dx : Sum (Buf .bf16) (Buf .f32)

/-- Element `j` of row `t` of the `[T, D]` view: `t·D + j`. -/
def xIdx (d : Inst) (t j : Nat) : Nat := t * d.D + j

/-- The elements of `y`, `dy` the launch reads and of `dx` it writes: the first `T·D`. -/
def rowSet (d : Inst) : Set Nat := {k | k < d.T * d.D}

/-- The elements of `rstd` the launch reads: the first `T`. -/
def rSet (d : Inst) : Set Nat := {t | t < d.T}

/-- The bytes of the elements `S` of a buffer of either dtype. -/
def elemsOf (b : Sum (Buf .bf16) (Buf .f32)) (S : Set Nat) : Set Int :=
  match b with
  | .inl b => b.elems S
  | .inr b => b.elems S

/-- The number of elements of a buffer of either dtype. -/
def lenOf (b : Sum (Buf .bf16) (Buf .f32)) : Nat :=
  match b with
  | .inl b => b.len
  | .inr b => b.len

/-- A buffer of either dtype is valid in memory `m`. -/
def validOf (b : Sum (Buf .bf16) (Buf .f32)) (m : Mem) : Prop :=
  match b with
  | .inl b => b.Valid m
  | .inr b => b.Valid m

/-- A buffer's first element, the kernel's argument. -/
def ptrOf (b : Sum (Buf .bf16) (Buf .f32)) : Sum (Ptr .bf16) (Ptr .f32) :=
  match b with
  | .inl b => .inl (b.ptr 0)
  | .inr b => .inr (b.ptr 0)

/-- Element `k` of a buffer of either dtype holds, in memory `m`, a finite value whose real value
is `v`. -/
def ValAt (b : Sum (Buf .bf16) (Buf .f32)) (m : Mem) (k : Nat) (v : ℝ) : Prop :=
  match b with
  | .inl b => ∃ c : BF16, m.read .bf16 (b.ptr k).addr = some c ∧ c.isFinite = true ∧ c.toReal = v
  | .inr b => ∃ c : F32, m.read .f32 (b.ptr k).addr = some c ∧ c.isFinite = true ∧ c.toReal = v

/-- Element `t` of `rstd` holds, in memory `m`, a finite `float32` whose real value is `r`. -/
def RstdAt (d : Inst) (m : Mem) (t : Nat) (r : ℝ) : Prop :=
  ∃ c : F32, m.read .f32 (d.rstd.ptr t).addr = some c ∧ c.isFinite = true ∧ c.toReal = r

/-- The instances covered: every `D` of this path (`1 … 512`), every `BD` Triton accepts with
`D ≤ BD ≤ 512` (the host's `next_power_of_2(D)` among them), every power-of-two `BT` of a block
Triton accepts (the host's `16`, and every `BT` of fla's autotune list), every `T` whose grid CUDA
accepts, every `eps`, and every dtype of `y`, `dy`, `dx` among `bfloat16` and `float32`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `D ≥ 1`: a tensor's last dimension, on this path. -/
  d_pos : 0 < d.D
  /-- `D ≤ 512`: this kernel's path (the host asserts it; fla launches another kernel beyond). -/
  d_le : d.D ≤ 512
  /-- `BD` is a power of two (Triton requires it of `tl.arange`). -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  /-- `D ≤ BD`: the row fits in the block (the host's `BD = next_power_of_2(D)`). -/
  d_bd : d.D ≤ d.BD
  /-- `BD ≤ 512`: the host's `BD` for `D ≤ 512`. -/
  bd_le : d.BD ≤ 512
  /-- `BT` is a power of two (Triton requires it of `tl.arange`). -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  /-- The block `[BT, BD]` has at most `2^20` elements (Triton's largest block). -/
  block_le : d.BT * d.BD ≤ 2 ^ 20
  /-- The grid `cdiv(T, BT)` has fewer than `2^31` programs (CUDA's limit on axis 0; a larger grid
  is rejected at launch). With `BT ≤ 2^20` it also gives `T < 2^51`, inside `int64`. -/
  grid_lt : (d.T + d.BT - 1) / d.BT < 2 ^ 31
  /-- `y` holds the `T·D` elements the launch reads. -/
  y_len : d.T * d.D ≤ lenOf d.y
  /-- `dy` holds the `T·D` elements the launch reads. -/
  dy_len : d.T * d.D ≤ lenOf d.dy
  /-- `dx` holds the `T·D` elements the launch writes. -/
  dx_len : d.T * d.D ≤ lenOf d.dx
  /-- `rstd` holds the `T` elements the launch reads. -/
  r_len : d.T ≤ d.rstd.len
  /-- The elements of `dx` the launch writes share no byte with the elements of `y` it reads (the
  host allocates `dx` fresh). -/
  dx_y : Disjoint (elemsOf d.dx (rowSet d)) (elemsOf d.y (rowSet d))
  /-- The elements of `dx` the launch writes share no byte with the elements of `dy` it reads. -/
  dx_dy : Disjoint (elemsOf d.dx (rowSet d)) (elemsOf d.dy (rowSet d))
  /-- The elements of `dx` the launch writes share no byte with the elements of `rstd` it
  reads. -/
  dx_r : Disjoint (elemsOf d.dx (rowSet d)) (d.rstd.elems (rSet d))

/-- The model on an instance: the buffers' first elements, `eps` as the `float32` the device gets
(`FloatV.hostF32` of the double), `T` as the runtime scalar Triton passes (`int32` below `2^31`,
`int64` from `2^31`), and the constexprs. -/
def model (d : Inst) : Kernel :=
  l2norm_bwd_kernel (ptrOf d.y) (d.rstd.ptr 0) (ptrOf d.dy) (ptrOf d.dx) (FloatV.hostF32 d.eps)
    (if d.T < 2 ^ 31 then Sum.inl (IntV.ofNat d.T) else Sum.inr (IntV.ofNat d.T))
    d.D d.BD d.NB d.BT

/-- The host's launch (source line 93): the grid `(triton.cdiv(T, BT),)`, `(T + BT − 1) / BT`. The
source's block shapes: `[BT]` (`tl.arange(0, BT)`, `o_t`, `m_t`, `p_rstd`, `b_rstd`, the row
sums `s`), `[BD]` (`o_d`), `[BT, 1]` (`o_t[:, None]`, `m_t[:, None]`, the product with `D` and its
pointers, `b_rstd[:, None]`, `s[:, None]`), `[1, BD]` (`o_d[None, :]` and its comparison) and
`[BT, BD]` (`m_x`, `p_y`, `p_dy`, `p_dx`, the loaded tiles and their widenings, `t1`, `pr`, `t2`,
`t3`, `b_dx`, and `b_dx16` for a `bfloat16` `dx`). The launch passes `num_warps = 4` and no other
option, so `enable_fp_fusion` is Triton's default `true`, `debug` is false, `num_ctas = 1`, no
`launch_pdl`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.T + d.BT - 1) / d.BT)
    [[d.BT], [d.BD], [d.BT, 1], [1, d.BD], [d.BT, d.BD]] (fusion := true)

/-- The exact dot product of a row, `S = Σ_{j<D} g_j·a_j`, in the reals. -/
noncomputable def rowDot (a g : Nat → ℝ) (D : Nat) : ℝ := ∑ j ∈ Finset.range D, g j * a j

/-- The absolute dot product of a row, `A = Σ_{j<D} |g_j·a_j|`, in the reals. -/
noncomputable def rowAbs (a g : Nat → ℝ) (D : Nat) : ℝ := ∑ j ∈ Finset.range D, |g j * a j|

/-- The inputs covered: the four buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); every element of `y`, `dy` and `rstd` the launch reads is finite; and on every row `t`,
with `a`, `g` its `y` and `dy` values and `r` its `rstd`,
`(1 + |r|)·(1 + |a_j|)·(1 + |g_j| + A) ≤ 2^124` for each `j < D` (room below `float32`'s
overflow; README, Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  validOf d.y init ∧ validOf d.dy init ∧ validOf d.dx init ∧ d.rstd.Valid init ∧
    (∀ t, t < d.T → (∃ r, RstdAt d init t r) ∧
      ∀ j, j < d.D → (∃ v, ValAt d.y init (xIdx d t j) v) ∧ (∃ v, ValAt d.dy init (xIdx d t j) v)) ∧
    ∀ t, t < d.T → ∀ (a g : Nat → ℝ) (r : ℝ),
      (∀ j, j < d.D → ValAt d.y init (xIdx d t j) (a j)) →
      (∀ j, j < d.D → ValAt d.dy init (xIdx d t j) (g j)) → RstdAt d init t r →
      ∀ j, j < d.D → (1 + |r|) * (1 + |a j|) * (1 + |g j| + rowAbs a g d.D) ≤ 2 ^ 124

/-! ## The specification -/

/-- The relative tolerance of a `float32` `dx[t, j]`, `2^-18`, relative to `M_j` (README,
Tolerances). -/
def dxTol32 : ℚ := 2 ^ (-18 : ℤ)

/-- The absolute floor of a `float32` `dx[t, j]`, `2^-138`, times `F_j` (README, Tolerances). -/
def dxFloor32 : ℚ := 2 ^ (-138 : ℤ)

/-- The rounding term of a `bfloat16` `dx[t, j]`, `2^-8` relative to `|ref_j|` (one `bfloat16`
rounding; README, Tolerances). -/
def dxRnd16 : ℚ := 2 ^ (-8 : ℤ)

/-- The relative tolerance of a `bfloat16` `dx[t, j]`, `2^-17`, relative to `M_j` (README,
Tolerances). -/
def dxTol16 : ℚ := 2 ^ (-17 : ℤ)

/-- The absolute floor of a `bfloat16` `dx[t, j]`, `2^-133`, times `F_j` (README, Tolerances). -/
def dxFloor16 : ℚ := 2 ^ (-133 : ℤ)

/-- Element `k` of `dx` in the final memory is a finite value of `dx`'s dtype within the tolerance
of that dtype of `ref` (with the scale `M` and the floor factor `F`):
`dxRnd16·|ref| + dxTol16·M + dxFloor16·F` for `bfloat16`, `dxTol32·M + dxFloor32·F` for
`float32`. -/
def DxOk (d : Inst) (fin : Mem) (k : Nat) (ref M F : ℝ) : Prop :=
  match d.dx with
  | .inl b => ∃ c : BF16, fin.read .bf16 (b.ptr k).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - ref| ≤ (dxRnd16 : ℝ) * |ref| + (dxTol16 : ℝ) * M + (dxFloor16 : ℝ) * F
  | .inr b => ∃ c : F32, fin.read .f32 (b.ptr k).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - ref| ≤ (dxTol32 : ℝ) * M + (dxFloor32 : ℝ) * F

/-- What the launch leaves: for each row `t < T`, with `a j`, `g j` the row's `y` and `dy` values
and `r` its `rstd` in the initial memory, each `dx[t, j]`, `j < D`, is a finite value of `dx`'s
dtype within its tolerance (`DxOk`) of the exact `g_j·r − S·a_j·r`, `S = Σ_k g_k·a_k`, with the
scale `M_j = |r|·(|g_j| + A·|a_j|)`, `A = Σ_k |g_k·a_k|`, and the floor factor
`F_j = (1 + |r|)·(1 + |a_j|)`; every byte outside the elements of `dx` the launch writes is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.T → ∃ (a g : Nat → ℝ) (r : ℝ),
      (∀ j, j < d.D → ValAt d.y init (xIdx d t j) (a j)) ∧
      (∀ j, j < d.D → ValAt d.dy init (xIdx d t j) (g j)) ∧ RstdAt d init t r ∧
      ∀ j, j < d.D → DxOk d fin (xIdx d t j)
        (g j * r - rowDot a g d.D * a j * r)
        (|r| * (|g j| + rowAbs a g d.D * |a j|))
        ((1 + |r|) * (1 + |a j|))) ∧
    ∀ z, z ∉ elemsOf d.dx (rowSet d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `l2norm_bwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `1e-6` (fla's default `eps`), pattern `0x3EB0C6F7A0B5ED8D`. -/
def eps1e6 : F64 := FloatV.ofPat 0x3EB0C6F7A0B5ED8D

/-- A buffer of `n` elements at byte address `base`: `bfloat16`, or `float32` when `f32`. -/
def bufAt (base n : Nat) (f32 : Bool) : Sum (Buf .bf16) (Buf .f32) :=
  if f32 then .inr ⟨base, n⟩ else .inl ⟨base, n⟩

/-- A launch of the host on `T` rows of `D` elements (`BD = next_power_of_2(D)`): `BT = 16`,
`NB = cdiv(T, 2048·32)`, `eps = 1e-6`, `y`, `dy`, `dx`, `rstd` at the byte addresses `2^32`,
`2^33`, `2^34`, `2^35`; `y` and `dx` are `float32` when `yf32` (`output_dtype = torch.float32` in
the forward), `bfloat16` otherwise, and `dy` is `float32` when `dyf32`. -/
def mk (T D BD : Nat) (yf32 dyf32 : Bool := false) : Inst where
  T := T
  D := D
  BD := BD
  NB := (T + 65535) / 65536
  BT := 16
  eps := eps1e6
  y := bufAt (2 ^ 32) (T * D) yf32
  rstd := ⟨2 ^ 35, T⟩
  dy := bufAt (2 ^ 33) (T * D) dyf32
  dx := bufAt (2 ^ 34) (T * D) yf32

theorem bytes_disjoint {d₁ d₂ : DType} (b : Buf d₁) (c : Buf d₂)
    (h : b.base + b.len * d₁.bytes ≤ c.base ∨ c.base + c.len * d₂.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro z hz hz'
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hz hz'
  omega

theorem mk_ok (T D BD e : Nat) (hD : 0 < D) (hD' : D ≤ 512) (he : BD = 2 ^ e) (hDB : D ≤ BD)
    (hB : BD ≤ 512) (hT : T * D ≤ 2 ^ 28) (hT' : T ≤ 2 ^ 27) (yf32 dyf32 : Bool := false) :
    (mk T D BD yf32 dyf32).ok := by
  cases yf32 <;> cases dyf32 <;>
  refine ⟨hD, hD', ⟨e, he⟩, hDB, hB, ⟨4, rfl⟩, ?_, ?_, ?_, ?_, ?_, le_refl _, ?_, ?_, ?_⟩ <;>
  first
    | (show 16 * BD ≤ 2 ^ 20; omega)
    | (show (T + 16 - 1) / 16 < 2 ^ 31; omega)
    | (simp only [mk, bufAt, lenOf]; simp; done)
    | (simp only [mk, bufAt, elemsOf, ite_true, ite_false, Bool.false_eq_true]
       refine Buf.elems_disjoint_of _ _ (bytes_disjoint _ _ ?_) (fun j hj => hj) (fun j hj => hj)
       first
         | left
           simp only [DType.bytes]
           push_cast
           omega
         | right
           simp only [DType.bytes]
           push_cast
           omega)

/-! ## Coverage: the host's test cases (source lines 109–119, `(B, T, H, D)`, `T = B·T·H` rows)
and the configurations SETUP.md names -/

/-- `(1, 1, 1, 1)`: one row, `D = 1`. -/
def setup_1_1_1_1 : Inst := mk 1 1 1
theorem setup_1_1_1_1_ok : setup_1_1_1_1.ok := mk_ok 1 1 1 0 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 1, 1, 128)`. -/
def setup_1_1_1_128 : Inst := mk 1 128 128
theorem setup_1_1_1_128_ok : setup_1_1_1_128.ok := mk_ok 1 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 63, 1, 60)`: `BD = 64`. -/
def setup_1_63_1_60 : Inst := mk 63 60 64
theorem setup_1_63_1_60_ok : setup_1_63_1_60.ok := mk_ok 63 60 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 65, 2, 512)`: `D = 512`, the largest on this path. -/
def setup_1_65_2_512 : Inst := mk 130 512 512
theorem setup_1_65_2_512_ok : setup_1_65_2_512.ok := mk_ok 130 512 512 9 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 500, 4, 64)`. -/
def setup_2_500_4_64 : Inst := mk 4000 64 64
theorem setup_2_500_4_64_ok : setup_2_500_4_64.ok := mk_ok 4000 64 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 1000, 2, 100)`: `BD = 128`. -/
def setup_2_1000_2_100 : Inst := mk 4000 100 128
theorem setup_2_1000_2_100_ok : setup_2_1000_2_100.ok := mk_ok 4000 100 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(3, 1024, 4, 128)`. -/
def setup_3_1024_4_128 : Inst := mk 12288 128 128
theorem setup_3_1024_4_128_ok : setup_3_1024_4_128.ok := mk_ok 12288 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 333, 3, 256)`. -/
def setup_2_333_3_256 : Inst := mk 1998 256 256
theorem setup_2_333_3_256_ok : setup_2_333_3_256.ok := mk_ok 1998 256 256 8 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(4, 4096, 16, 128)`: DeltaNet's 16 heads of 128 on a training batch of 4 × 4096 tokens
(SETUP.md), `2^18` rows. -/
def setup_4_4096_16_128 : Inst := mk 262144 128 128
theorem setup_4_4096_16_128_ok : setup_4_4096_16_128.ok := mk_ok 262144 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- GatedDeltaNet's 6 heads of 256 on a training batch of 4 × 4096 tokens (SETUP.md). -/
def setup_gdn_4_4096_6_256 : Inst := mk 98304 256 256
theorem setup_gdn_4_4096_6_256_ok : setup_gdn_4_4096_6_256.ok := mk_ok 98304 256 256 8 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- PaTH attention's `w = l2_norm(w, output_dtype=torch.float32)` (SETUP.md): 32 heads of 64 on a
training batch of 4 × 4096 tokens, `y`, `dy` and `dx` of `float32`. -/
def setup_path_4_4096_32_64_f32 : Inst := mk 524288 64 64 true true
theorem setup_path_4_4096_32_64_f32_ok : setup_path_4_4096_32_64_f32.ok := mk_ok 524288 64 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) true true

/-- A `float32` gradient into a `bfloat16` `y` (SETUP.md): 16 heads of 128 on 4 × 4096 tokens,
`dx` of `bfloat16`. -/
def setup_4_4096_16_128_dyf32 : Inst := mk 262144 128 128 false true
theorem setup_4_4096_16_128_dyf32_ok : setup_4_4096_16_128_dyf32.ok := mk_ok 262144 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) false true

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `bfloat16` read of `zeroMem` gives `+0`. -/
theorem zeroMem_read16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    zeroMem.read .bf16 a = some (FloatV.ofPat 0 : BF16) := by
  have hb : zeroMem.readBytes a 2 = some [0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [ha, ite_true, hb]
  decide

/-- Every aligned `float32` read of `zeroMem` gives `+0`. -/
theorem zeroMem_read32 {a : Int} (ha : Mem.Aligned .f32 a) :
    zeroMem.read .f32 a = some (FloatV.ofPat 0 : F32) := by
  have hb : zeroMem.readBytes a 4 = some [0, 0, 0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem bf16_zero_toReal : (FloatV.ofPat 0 : BF16).toReal = 0 := by
  show ((Format.bfloat16.toRat 0 : ℚ) : ℝ) = 0
  have : Format.bfloat16.toRat 0 = 0 := by decide +kernel
  rw [this]; simp

theorem f32_zero_toReal : (FloatV.ofPat 0 : F32).toReal = 0 := by
  show ((Format.binary32.toRat 0 : ℚ) : ℝ) = 0
  have : Format.binary32.toRat 0 = 0 := by decide +kernel
  rw [this]; simp

theorem claim_satisfiable : claim.Satisfiable := by
  have hal16 : ∀ k : Nat, k < 1 → Mem.Aligned .bf16 ((⟨2 ^ 32, 1⟩ : Buf .bf16).ptr k).addr ∧
      Mem.Aligned .bf16 ((⟨2 ^ 33, 1⟩ : Buf .bf16).ptr k).addr := by
    intro k hk
    have : k = 0 := by omega
    subst this
    constructor <;> (unfold Mem.Aligned; simp [Buf.ptr_addr, DType.bytes])
  have hal32 : Mem.Aligned .f32 ((⟨2 ^ 35, 1⟩ : Buf .f32).ptr 0).addr := by
    unfold Mem.Aligned; simp [Buf.ptr_addr, DType.bytes]
  have hy : ∀ k : Nat, k < 1 → ValAt setup_1_1_1_1.y zeroMem k 0 := fun k hk =>
    ⟨FloatV.ofPat 0, zeroMem_read16 (hal16 k hk).1, by decide, bf16_zero_toReal⟩
  have hdy : ∀ k : Nat, k < 1 → ValAt setup_1_1_1_1.dy zeroMem k 0 := fun k hk =>
    ⟨FloatV.ofPat 0, zeroMem_read16 (hal16 k hk).2, by decide, bf16_zero_toReal⟩
  have hr : RstdAt setup_1_1_1_1 zeroMem 0 0 :=
    ⟨FloatV.ofPat 0, zeroMem_read32 hal32, by decide, f32_zero_toReal⟩
  have hidx : ∀ j, j < setup_1_1_1_1.D → xIdx setup_1_1_1_1 0 j < 1 := by
    intro j hj; simp [xIdx, setup_1_1_1_1, mk] at hj ⊢; omega
  refine ⟨setup_1_1_1_1, zeroMem, setup_1_1_1_1_ok, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk, bufAt], by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk, bufAt], by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk, bufAt], by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, bufAt, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk], by norm_num [setup_1_1_1_1, mk, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · intro t ht
    have ht0 : t = 0 := by simp [setup_1_1_1_1, mk] at ht; omega
    subst ht0
    exact ⟨⟨0, hr⟩, fun j hj => ⟨⟨0, hy _ (hidx j hj)⟩, ⟨0, hdy _ (hidx j hj)⟩⟩⟩
  · intro t ht a g r ha hg hr' j hj
    have ht0 : t = 0 := by simp [setup_1_1_1_1, mk] at ht; omega
    have hj0 : j = 0 := by simp [setup_1_1_1_1, mk] at hj; omega
    subst ht0; subst hj0
    have hD : setup_1_1_1_1.D = 1 := rfl
    have hval : ∀ {b : Sum (Buf .bf16) (Buf .f32)} {k : Nat} {v w : ℝ},
        ValAt b zeroMem k v → ValAt b zeroMem k w → v = w := by
      intro b k v w hv hw
      cases b with
      | inl b =>
        obtain ⟨c, hc, -, rfl⟩ := hv; obtain ⟨c', hc', -, rfl⟩ := hw
        rw [hc] at hc'; cases hc'; rfl
      | inr b =>
        obtain ⟨c, hc, -, rfl⟩ := hv; obtain ⟨c', hc', -, rfl⟩ := hw
        rw [hc] at hc'; cases hc'; rfl
    have ha0 : a 0 = 0 := hval (ha 0 (by rw [hD]; norm_num)) (hy _ (hidx 0 (by rw [hD]; norm_num)))
    have hg0 : g 0 = 0 := hval (hg 0 (by rw [hD]; norm_num)) (hdy _ (hidx 0 (by rw [hD]; norm_num)))
    have hr0 : r = 0 := by
      obtain ⟨c, hc, -, rfl⟩ := hr'; obtain ⟨c', hc', -, h'⟩ := hr
      rw [hc] at hc'; cases hc'; exact h'
    simp only [rowAbs, hD, Finset.sum_range_one, ha0, hg0, hr0]
    norm_num

end Claims.L2NormBwd
