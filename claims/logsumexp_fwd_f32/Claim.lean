import TVA

/-!
# `logsumexp_fwd_f32`: block-wise log-sum-exp of rows, forward (fla)

The kernel is `logsumexp_fwd_kernel` of `source/ops.py` (lines 67–91), which calls the
`@triton.jit` helpers `exp` and `log` (lines 55–60); it reads the module-level names `triton` and
`tl` (lines 50–51). Its host `logsumexp_fwd` (lines 94–108) launches it on the grid `(N, ND)`,
`ND = cdiv(D, BD)`, `BD = min(next_power_of_2(D), 65536)`, on a contiguous `float32` `x` viewed as
`[N, D]` and a fresh `float32` `z` of `N·ND` elements, with `HAS_SOFTCAPPING = False` (no
softcapping) and `HAS_SCALE` either way; `SETUP.md` lists the launches. Program `(n, i)` reduces
the columns `[i·BD, min(D, (i+1)·BD))` of row `n` to one value.

What is claimed: every covered launch ends normally; every `z[n, i]` holds a `float32` value; when
the block's inputs are finite and `|y_j| ≤ 2^126` (no scale) or `|y_j| ≤ 2^20` (with a scale),
`y_j = x[n, j]·σ` (`σ` the host's double `scale`, or `1`), `z[n, i]` is finite and within
`zTolM·M + zTol0` (`2^-20`, `2^-16`) of the exact `log Σ_j e^(y_j)` over the block, for every
bound `M` of the block's `|y_j|`; every other byte is unchanged.

The conditions: `BD` a power of two at most `2^16`; `1 ≤ D < 2^31`; the grid within CUDA's limits
(`ND < 2^16`) and fewer than `2^31` programs; the buffers hold the `N·D` (`x`) and `N·ND` (`z`)
elements the launch touches, and the elements of `z` share no byte with those of `x`; with a
scale, `scale` is a finite double of magnitude at most `2^126`. Every `float32` input is
covered.

The tolerances rest on the device facts for `tl.exp` and `tl.log` on `float32` and on the depth
of a `tl.sum`'s tree (`Device`).
-/

namespace Claims.LogsumexpFwd

open TVA

/-! ## The model -/

/-- `-float('inf')`: the Python double `-∞` (pattern `0xFFF0000000000000`) as the `float32`
constant Triton builds for it (`litF32`, pattern `0xFF800000`). -/
def negInfC : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

set_option linter.unusedVariables false in
/-- `logsumexp_fwd_kernel` (source lines 67–91), with `HAS_SOFTCAPPING = False`. The pointers are
`*fp32` (the host passes a `float32` `x` and a `float32` `z`). `scale` is the runtime `float32`
scalar (`FloatV.hostF32` of the Python `float`) when `HAS_SCALE` is true; when it is false the
host passes `None`, a compile-time value the body never reads. `softcapping` is `None` (the
compile-time `if HAS_SOFTCAPPING` of line 86 is not visited). `D`, `BD` and the flags are
`tl.constexpr`s, Lean parameters; the flags' branches are compile-time `if`s, written as Lean
`if`s. The helpers `exp(x)` and `log(x)` (lines 55–60) are inlined: `tl.exp(x.to(tl.float32))`
and `tl.log(x.to(tl.float32))`, where `x` is already `float32`, so `.to` is `x` itself.

With a scale, `b_x * scale` is a product, and it feeds the subtraction of line 90 and, through
`tl.max`, the addition of line 90 (`b_m`): both are the fusable forms (`tl.subc`, `tl.addc`).
Without a scale no product feeds them: the IEEE `tl.sub`, `tl.add`. -/
def logsumexp_fwd_kernel (x z : Ptr .f32) (scale : F32) (D BD : Nat) (HAS_SCALE : Bool) :
    Kernel where
  body ctx := do
    -- line 78: `tl.program_id(0).to(tl.int64)`, `tl.program_id(1).to(tl.int64)`: axes 0 and 1
    -- of the grid, `int32` scalars, sign-extended to `int64`
    let i_n : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    let i_d : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
    -- line 79: `i_d * BD + tl.arange(0, BD)`: the constexpr `BD` built in `int64` (the dtype of
    -- the scalar tensor `i_d`), the `int64` product; the `int32` block `tl.arange(0, BD)`
    -- sign-extended to `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [BD] I64 ← tl.convert .i32 .i64 (tl.arange 0 BD)
    let o_d : Tensor [BD] I64 := i_d * (IntV.ofNat BD : I64) + o_a
    -- line 80: `o_d < D`: the constexpr `D` (below `2^31`) becomes an `int32` tensor, promoted to
    -- `int64` by sign extension (the same value), then the signed comparison
    let m_d : Tensor [BD] Bool := o_d.ltS (IntV.ofNat D : I64)
    -- line 83: `x + i_n * D + o_d`, left to right: the constexpr `D` built in `int64`, the `int64`
    -- product added to the `*fp32` pointer, then the `int64` block `o_d` added to it; the masked
    -- load with `other=-float('inf')`, a `float32` tensor (`negInfC`), already the pointer's
    -- dtype
    let b_x ← tl.load (x + i_n * (IntV.ofNat D : I64) + o_d) (some m_d) (some (bc negInfC))
    -- lines 84–85: `if HAS_SCALE: b_x = b_x * scale`, the IEEE `float32` product with the
    -- runtime scalar broadcast to the block
    let b_x ← if HAS_SCALE then tl.mul b_x (bc scale) else pure b_x
    -- line 88: `tl.where(m_d, b_x, -float('inf'))`: the Python `-inf`, a value of `tl.where`,
    -- built in the operation's dtype `float32` (`negInfC`)
    let b_x := tl.where m_d b_x (bc negInfC)
    -- line 89: `tl.max(b_x, 0)`: the maximum of the `float32` block (no widening)
    let b_m ← tl.maxAll true b_x
    -- line 90: `b_x - b_m`, the scalar broadcast to every lane: `tl.subc` with a scale (the
    -- product feeds it), `tl.sub` without; then `exp`, the platform's `tl.exp` on `float32`
    let b_t ← if HAS_SCALE then tl.subc b_x (bc b_m) else tl.sub b_x (bc b_m)
    let b_e ← tl.exp b_t
    -- `tl.sum(…, 0)`: its inputs are `tl.exp` results, not products: IEEE additions along a tree
    let b_s ← tl.sum b_e
    -- `log(…)`: the platform's `tl.log` on the `float32` scalar
    let b_l := (← tl.log (bc b_s : Tensor [] F32)).item
    -- `… + b_m`: `tl.addc` with a scale (`b_m` is a maximum of products), `tl.add` without
    let b_z := (← if HAS_SCALE then tl.addc (bc b_l : Tensor [] F32) (bc b_m)
      else tl.add (bc b_l : Tensor [] F32) (bc b_m)).item
    -- line 91: `z + i_n * tl.cdiv(D, BD) + i_d`: `tl.cdiv` of two compile-time integers is
    -- Python's (`tl.py.cdiv`), built in `int64` beside `i_n`, the `int64` product added to the
    -- `*fp32` pointer, then the `int64` scalar `i_d`; `b_z` is already `float32`, the pointer's
    -- dtype: the store of one element
    tl.store1 (z + i_n * (IntV.ofInt (tl.py.cdiv D BD) : I64) + i_d) b_z

/-! ## The conditions -/

/-- What the host chooses: one launch of `logsumexp_fwd_kernel` through `logsumexp_fwd(x, scale,
softcapping=None)` (or a direct launch with the same arguments and any power-of-two `BD`). -/
structure Inst where
  /-- `N`: the rows of the view `x.view(-1, x.shape[-1])`. -/
  N : Nat
  /-- `D`: `x.shape[-1]`, the row length. -/
  D : Nat
  /-- `BD`: the host's `min(triton.next_power_of_2(D), 64 * 1024)`. -/
  BD : Nat
  /-- `HAS_SCALE = scale is not None`. -/
  hasScale : Bool
  /-- `scale` (the Python `float`, a double; unused when `hasScale` is false). -/
  scale : F64
  /-- `x`: the input, `float32`, contiguous; `x[n, j]` is element `n·D + j`. -/
  x : Buf .f32
  /-- `z`: the output, `float32`, `[N, ND]`; `z[n, i]` is element `n·ND + i`. -/
  z : Buf .f32

/-- `ND = triton.cdiv(D, BD)`, the blocks of a row (the grid's axis 1). -/
def ND (d : Inst) : Nat := (d.D + d.BD - 1) / d.BD

/-- Element `j` of row `n` of `x`: `n·D + j`. -/
def xIdx (d : Inst) (n j : Nat) : Nat := n * d.D + j

/-- Element `i` of row `n` of `z`: `n·ND + i`. -/
def zIdx (d : Inst) (n i : Nat) : Nat := n * ND d + i

/-- The elements of `x` the launch reads: the first `N·D`. -/
def xSet (d : Inst) : Set Nat := {k | k < d.N * d.D}

/-- The elements of `z` the launch writes: the first `N·ND`. -/
def zSet (d : Inst) : Set Nat := {k | k < d.N * ND d}

/-- The instances covered: every row length `D` below `2^31` with every power-of-two `BD` up to
`2^16` (the host's `min(next_power_of_2(D), 65536)` among them), every `N` whose grid CUDA and
Triton's launcher accept, and every finite scale of magnitude at most `2^126`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `D ≥ 1`: a tensor's last dimension (the host's `cdiv(D, BD)` divides by `BD = 0` at
  `D = 0`). -/
  d_pos : 0 < d.D
  /-- `D < 2^31`: the constexpr `D` is an `int32`-range constant (README, Exclusions). -/
  d_lt : d.D < 2 ^ 31
  /-- `BD` is a power of two (Triton requires it of `tl.arange`). -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  /-- `BD ≤ 2^16`: the host's `min(…, 64 * 1024)`; the tolerances use it. -/
  bd_le : d.BD ≤ 2 ^ 16
  /-- `ND < 2^16`: CUDA's limit on the grid's axis 1 (a larger grid is rejected at launch). -/
  nd_lt : ND d < 2 ^ 16
  /-- Fewer than `2^31` programs: Triton's launcher skips a grid of `2^31` programs or more
  without an error (README, Exclusions). It also gives `N < 2^31`, CUDA's limit on axis 0. -/
  grid_lt : d.N * ND d < 2 ^ 31
  /-- `x` holds the `N·D` elements the launch reads. -/
  x_len : d.N * d.D ≤ d.x.len
  /-- `z` holds the `N·ND` elements the launch writes. -/
  z_len : d.N * ND d ≤ d.z.len
  /-- The elements of `z` the launch writes share no byte with the elements of `x` it reads (the
  host allocates `z` fresh). -/
  z_x : Disjoint (d.z.elems (zSet d)) (d.x.elems (xSet d))
  /-- With a scale: a finite double of magnitude at most `2^126` (its `float32` is finite; real
  logit scales are near `1`). -/
  scale_ok : d.hasScale = true → d.scale.isFinite = true ∧ |d.scale.toRat| ≤ 2 ^ 126

/-- The model on an instance: the buffers' first elements, the scale as the `float32` the device
gets (`FloatV.hostF32` of the double; any value when `hasScale` is false, never read), and the
constexprs. -/
def model (d : Inst) : Kernel :=
  logsumexp_fwd_kernel (d.x.ptr 0) (d.z.ptr 0) (FloatV.hostF32 d.scale) d.D d.BD d.hasScale

/-- The host's launch (source line 103): the grid `(N, ND, 1)`. The source's only block shape is
`[BD]` (`tl.arange(0, BD)`, `o_d`, `m_d`, the pointers, `b_x` and its successors before the
sum). The launch passes no option, so `enable_fp_fusion` is Triton's default `true`, `debug` is
false, `num_ctas = 1`, no `launch_pdl`. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.N, ND d, 1], fusion := true, blocks := [[d.BD]] }

/-- The inputs covered: both buffers are valid (aligned, inside `[0, 2^64)`, every byte present).
Every `float32` input value is covered, NaN and infinities included: `Post` constrains a block
whose values are not `BlockOk` less. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.z.Valid init

/-! ## The specification -/

/-- The columns of block `i`: `[i·BD, min(D, (i+1)·BD))`. -/
def blk (d : Inst) (i : Nat) : Finset Nat := Finset.Ico (i * d.BD) (min d.D ((i + 1) * d.BD))

/-- The factor applied to the inputs: the host's double `scale`, or `1` without a scale. -/
noncomputable def sig (d : Inst) : ℝ := if d.hasScale then (d.scale.toRat : ℝ) else 1

/-- The bound on the block's `|y_j|` the claim's accuracy needs: `2^126` without a scale (room in
`float32`), `2^20` with one (README, Exclusions). -/
noncomputable def yMax (d : Inst) : ℝ := if d.hasScale then 2 ^ 20 else 2 ^ 126

/-- A block whose log-sum-exp the claim bounds: its inputs `xs j` are finite `float32` values and
each `|y_j| = |xs j · σ|` is at most `yMax`. -/
def BlockOk (d : Inst) (xs : Nat → F32) (i : Nat) : Prop :=
  ∀ j ∈ blk d i, (xs j).isFinite = true ∧ |(xs j).toReal * sig d| ≤ yMax d

/-- The exact log-sum-exp of block `i` of a row `xs`: `log Σ_{j ∈ blk i} e^(xs j · σ)`, in the
reals. -/
noncomputable def lseRef (d : Inst) (xs : Nat → F32) (i : Nat) : ℝ :=
  Real.log (∑ j ∈ blk d i, Real.exp ((xs j).toReal * sig d))

/-- The tolerance per unit of the block's largest `|y_j|`: `2^-20` (the roundings of the scale,
of `x·scale`, of `y_j − m` and of `log + m`, each relative to the magnitudes; README,
Tolerances). -/
def zTolM : ℚ := 2 ^ (-20 : ℤ)

/-- The absolute tolerance: `2^-16` (`tl.exp`'s and `tl.log`'s errors and the sum's roundings,
over at most `2^16` lanes; README, Tolerances). -/
def zTol0 : ℚ := 2 ^ (-16 : ℤ)

/-- What the launch leaves: for each row `n < N` and block `i < ND`, with `xs` the row's input
values as read in the initial memory, `z[n, i]` holds a `float32` value; when the block is
`BlockOk`, that value is finite and, for every bound `M` of the block's `|xs j · σ|`, within
`zTolM·M + zTol0` of `lseRef`; every byte outside the elements of `z` the launch writes is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ n, n < d.N → ∀ i, i < ND d → ∃ xs : Nat → F32,
      (∀ j, j < d.D → init.read .f32 (d.x.ptr (xIdx d n j)).addr = some (xs j)) ∧
      ∃ c : F32, fin.read .f32 (d.z.ptr (zIdx d n i)).addr = some c ∧
        (BlockOk d xs i → c.isFinite = true ∧
          ∀ M : ℝ, (∀ j ∈ blk d i, |(xs j).toReal * sig d| ≤ M) →
            |c.toReal - lseRef d xs i| ≤ (zTolM : ℝ) * M + (zTol0 : ℝ))) ∧
    ∀ b, b ∉ d.z.elems (zSet d) → fin b = init b

/-! ## The claim -/

/-- **The claim**: every covered launch of `logsumexp_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The host's launch on `N` rows of `D` columns with `BD = 2^e`, the scale flag and the scale's
double pattern: `x` at `2^34`, a fresh `z` at `2^40`. -/
def mk (N D e : Nat) (hs : Bool) (sbits : Nat) : Inst where
  N := N
  D := D
  BD := 2 ^ e
  hasScale := hs
  scale := FloatV.ofPat sbits
  x := ⟨2 ^ 34, N * D⟩
  z := ⟨2 ^ 40, N * ((D + 2 ^ e - 1) / 2 ^ e)⟩

/-- Two element sets of buffers whose byte ranges do not meet share no byte. -/
theorem elems_disj {b c : Buf .f32} (S T : Set Nat)
    (h : b.base + b.len * 4 ≤ c.base) (hS : ∀ k ∈ S, k < b.len) (hT : ∀ k ∈ T, k < c.len) :
    Disjoint (b.elems S) (c.elems T) := by
  rw [Set.disjoint_left]
  rintro y ⟨j, hj, hy⟩ ⟨k, hk, hy'⟩
  have h1 := hS j hj
  have h2 := hT k hk
  simp only [Mem.span, Buf.ptr_addr, DType.bytes, Set.mem_ofPred_eq] at hy hy'
  push_cast at hy hy'
  have h1' : ((j : Nat) : Int) + 1 ≤ (b.len : Int) := by exact_mod_cast h1
  have h2' : (0 : Int) ≤ (k : Int) := by positivity
  nlinarith

/-- Every such launch is covered when `1 ≤ D < 2^31`, `e ≤ 16`, the grid has fewer than `2^31`
programs and fewer than `2^16` blocks per row, the matrix fits below `2^40` bytes, and the scale
(when there is one) is in range. -/
theorem mk_ok {N D e : Nat} {hs : Bool} {sbits : Nat} (hD : 0 < D) (hD31 : D < 2 ^ 31)
    (he : e ≤ 16) (hnd : (D + 2 ^ e - 1) / 2 ^ e < 2 ^ 16)
    (hg : N * ((D + 2 ^ e - 1) / 2 ^ e) < 2 ^ 31) (hsz : 2 ^ 34 + N * D * 4 ≤ 2 ^ 40)
    (hsc : hs = true → (FloatV.ofPat sbits : F64).isFinite = true ∧
      |(FloatV.ofPat sbits : F64).toRat| ≤ 2 ^ 126) :
    (mk N D e hs sbits).ok := by
  refine ⟨hD, hD31, ⟨e, rfl⟩, Nat.pow_le_pow_right (by norm_num) he, hnd, hg, le_refl _, le_refl _,
    ?_, hsc⟩
  refine Disjoint.symm (elems_disj _ _ ?_ (fun k hk => hk) (fun k hk => hk))
  simp only [mk]
  push_cast
  have : (2 : Int) ^ 34 + ((N * D : Nat) : Int) * 4 ≤ 2 ^ 40 := by exact_mod_cast hsz
  push_cast at this
  linarith

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 2 1025 11 false 0, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by simp), ?_⟩
  exact ⟨⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [zeroMem]⟩,
    ⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [zeroMem]⟩⟩

/-! ## Coverage of the host's launches (`SETUP.md`)

`logsumexp_fwd(x, scale)` on a contiguous `float32` `[N, D]`: `BD = min(next_power_of_2(D),
65536)`. The scale `1.0` is the double pattern `0x3FF0000000000000` (`4607182418800017408`),
`0.0` is `0`, `0.5` is `0x3FE0000000000000` (`4602678819172646912`) and `30.0` is
`0x403E000000000000` (`4629137466983448576`). Names: `setup_<N>x<D>_<noscale|scale…>`. -/

/-- The scale `1.0` is in range. -/
theorem one_ok : (FloatV.ofPat 4607182418800017408 : F64).isFinite = true ∧
    |(FloatV.ofPat 4607182418800017408 : F64).toRat| ≤ 2 ^ 126 :=
  ⟨by decide, by decide +kernel⟩

/-- The scale `0.0` is in range. -/
theorem zero_ok : (FloatV.ofPat 0 : F64).isFinite = true ∧
    |(FloatV.ofPat 0 : F64).toRat| ≤ 2 ^ 126 :=
  ⟨by decide, by decide +kernel⟩

/-- The scale `0.5` is in range. -/
theorem half_ok : (FloatV.ofPat 4602678819172646912 : F64).isFinite = true ∧
    |(FloatV.ofPat 4602678819172646912 : F64).toRat| ≤ 2 ^ 126 :=
  ⟨by decide, by decide +kernel⟩

/-- The scale `30.0` is in range. -/
theorem thirty_ok : (FloatV.ofPat 4629137466983448576 : F64).isFinite = true ∧
    |(FloatV.ofPat 4629137466983448576 : F64).toRat| ≤ 2 ^ 126 :=
  ⟨by decide, by decide +kernel⟩

def setup_1x1_noscale : Inst := mk 1 1 0 false 0
theorem setup_1x1_noscale_ok : setup_1x1_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

def setup_2x1025_noscale : Inst := mk 2 1025 11 false 0
theorem setup_2x1025_noscale_ok : setup_2x1025_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

def setup_2x1025_scale1 : Inst := mk 2 1025 11 true 4607182418800017408
theorem setup_2x1025_scale1_ok : setup_2x1025_scale1.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => one_ok)

def setup_2x1025_scale0 : Inst := mk 2 1025 11 true 0
theorem setup_2x1025_scale0_ok : setup_2x1025_scale0.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => zero_ok)

def setup_63x50257_noscale : Inst := mk 63 50257 16 false 0
theorem setup_63x50257_noscale_ok : setup_63x50257_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

def setup_64x65536_scale0p5 : Inst := mk 64 65536 16 true 4602678819172646912
theorem setup_64x65536_scale0p5_ok : setup_64x65536_scale0p5.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => half_ok)

def setup_32x128256_noscale : Inst := mk 32 128256 16 false 0
theorem setup_32x128256_noscale_ok : setup_32x128256_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

def setup_16x151936_scale1 : Inst := mk 16 151936 16 true 4607182418800017408
theorem setup_16x151936_scale1_ok : setup_16x151936_scale1.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => one_ok)

def setup_8x262144_scale30 : Inst := mk 8 262144 16 true 4629137466983448576
theorem setup_8x262144_scale30_ok : setup_8x262144_scale30.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => thirty_ok)

def setup_2048x32000_scale1 : Inst := mk 2048 32000 15 true 4607182418800017408
theorem setup_2048x32000_scale1_ok : setup_2048x32000_scale1.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => one_ok)

def setup_1024x128256_noscale : Inst := mk 1024 128256 16 false 0
theorem setup_1024x128256_noscale_ok : setup_1024x128256_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

def setup_512x151936_scale1 : Inst := mk 512 151936 16 true 4607182418800017408
theorem setup_512x151936_scale1_ok : setup_512x151936_scale1.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ => one_ok)

def setup_256x262144_noscale : Inst := mk 256 262144 16 false 0
theorem setup_256x262144_noscale_ok : setup_256x262144_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun h => by simp at h)

end Claims.LogsumexpFwd
