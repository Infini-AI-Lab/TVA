import TVA

/-!
# `swiglu_fwd_bf16`: SwiGLU, forward (fla)

The kernel is `swiglu_fwd_kernel` of `source/swiglu.py` (lines 92–102; it reads the module-level
names `triton`, its decorator, and `tl`, bound by the imports of lines 55–56). Its host
`swiglu_fwd` (lines 105–124) launches it on the grid `(triton.cdiv(T, B),)`, `T = x.numel()`, with
`D = x.shape[-1]`, on two `bfloat16` inputs `x`, `y` of one shape, each with unit stride along the
last dimension and one row stride for the leading dimensions (`prepare_input`), and a fresh
contiguous `bfloat16` output `z` (row stride `D`); `SETUP.md` lists the launches. Program `p`
handles the flat elements `k ∈ [p·B, p·B + B)`, `k < T`: element `k` is column `c = k % D` of row
`r = k // D`, and the kernel stores `z[r, c] = x[r, c] · sigmoid(x[r, c]) · y[r, c]`, computed in
`float32` and rounded to `bfloat16`.

What is claimed: every covered launch ends normally, and for every row `r < N` (`N = T / D`) and
column `c < D`, with `a = x[r, c]` and `b = y[r, c]` the initial inputs, `z[r, c]` is a finite
`bfloat16`, and
* for `a > -89`: within `zTol = (2^-8 + 2^-13)·|zRef| + 2^-133` of the exact
  `zRef = a · b / (1 + e^-a)`;
* for `a ≤ -89`: zero (`tl.exp(-a)` overflows to `+∞` and the kernel's sigmoid is `+0`);
every byte outside the first `N·D` elements of `z` is unchanged.

The conditions: `D ≥ 1`; `B` a power of two, at most `2^20`; fewer than `2^31` programs; the
integer arguments below `2^63` and `D` below `2^31`; the buffers hold every element the launch
touches, and the written elements of `z` share no byte with the elements of `x` and `y` read (the
host allocates `z`); every input read is a finite `bfloat16`, and every exact result is at most
`2^127` in magnitude (half of `bfloat16`'s largest value).

The tolerance rests on the device facts for `tl.exp` and `/` on `float32` (`Device.ExpF32`,
`Device.ExpF32Overflow`, `Device.DivFull`, `Device.DivF32InfDivisor`, and the totality facts);
README, Tolerances derives it.
-/

namespace Claims.SwigluFwd

open TVA

/-! ## The model -/

/-- An unannotated integer argument of the kernel as Triton passes it: an `int32` below `2^31`, an
`int64` from `2^31` (`Sum.inl`, `Sum.inr`; `argOf` picks by the value). -/
def argOf (n : Nat) : Sum I32 I64 :=
  if n < 2 ^ 31 then .inl (IntV.ofNat n) else .inr (IntV.ofNat n)

/-- Such an argument beside an `int64` operand (a product with, or a comparison against, an `int64`
block): an `int32` is sign-extended to `int64` (the operation's promotion), an `int64` is used as
it is. -/
def arg64 : Sum I32 I64 → Prog I64
  | .inl t => do pure (← tl.convert .i32 .i64 (Tensor.scalar t)).item
  | .inr t => pure t

/-- `swiglu_fwd_kernel` (source lines 92–102). `x`, `y` and `z` are `*bf16` pointers (the host's
`bfloat16` tensors, SETUP.md). `stride_x_row`, `stride_y_row`, `stride_z_row` and `T` are
unannotated integer arguments, each used only beside an `int64` block (`row * stride`, `offs < T`):
the model takes each as `Sum I32 I64` and promotes it with `arg64`. `T` is in `do_not_specialize`,
so it is never a compile-time constant. A stride equal to `1` is the compile-time constant `1`
(Triton's specialisation); as an operand of `row * 1` it is built in the product's dtype `int64`,
the value `arg64` gives, so the model is the same. `D` and `B` are `tl.constexpr`s, Lean
parameters. -/
def swiglu_fwd_kernel (x y z : Ptr .bf16) (stride_x_row stride_y_row stride_z_row T : Sum I32 I64)
    (D B : Nat) : Kernel where
  body ctx := do
    -- line 94: `i_n = tl.program_id(0).to(tl.int64)`: axis 0 of the grid, an `int32` scalar,
    -- sign-extended to `int64`
    let i_n : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 95: `i_n * B + tl.arange(0, B)`: the constexpr `B` built in `int64` (the dtype of the
    -- scalar tensor `i_n`), the `int64` product; the `int32` block `tl.arange(0, B)` sign-extended
    -- to `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [B] I64 ← tl.convert .i32 .i64 (tl.arange 0 B)
    let offs : Tensor [B] I64 := i_n * (IntV.ofNat B : I64) + o_a
    -- line 96: `offs < T`: `T` promoted to `int64`, the signed comparison
    let T64 ← arg64 T
    let mask : Tensor [B] Bool := offs.ltS T64
    -- lines 97-98: `int64` `//` and `%` (truncating) by the constexpr `D` built in `int64`
    let row ← tl.idiv offs (bc (IntV.ofNat D : I64))
    let col ← tl.irem offs (bc (IntV.ofNat D : I64))
    -- line 99: `x + row * stride_x_row + col`, left to right: the `int64` block `row` times the
    -- promoted stride, added to the `*bf16` pointer, then the `int64` block `col` added to those
    -- pointers; the masked load with `other=0.`: the Python `0.` is a `float32` tensor (`+0`),
    -- converted to the pointer's dtype, `+0` in `bfloat16` (exact; the value `litBF16` of `0`
    -- gives); then `.to(tl.float32)`, the exact widening (`tl.cvt`)
    let sx ← arg64 stride_x_row
    let b_x16 : Tensor [B] BF16 ←
      tl.load (x + row * sx + col) (some mask) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_x : Tensor [B] F32 ← tl.cvt b_x16
    -- line 100: the same for `y`
    let sy ← arg64 stride_y_row
    let b_y16 : Tensor [B] BF16 ←
      tl.load (y + row * sy + col) (some mask) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_y : Tensor [B] F32 ← tl.cvt b_y16
    -- line 101: `b_x * tl.sigmoid(b_x) * b_y`, left to right. `tl.sigmoid` is Triton's body
    -- `1 / (1 + math.exp(-x))` (`tl.sigmoid`, `TVA.Triton.Arith`: the negation, `tl.exp`, the IEEE
    -- `+` of the constant `1`, the default `float32` division of `1` by the sum); then two IEEE
    -- `float32` products (no addition follows either, so nothing fuses)
    let s ← tl.sigmoid b_x
    let xs ← tl.mul b_x s
    let b_z ← tl.mul xs b_y
    -- line 102: `b_z.to(z.dtype.element_ty)`: to `bfloat16`, one rounding to nearest even
    -- (`tl.cvt`); the pointers as on line 99; the masked store
    let sz ← arg64 stride_z_row
    let b_z16 : Tensor [B] BF16 ← tl.cvt b_z
    tl.store (z + row * sz + col) b_z16 (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `swiglu_fwd_kernel`. -/
structure Inst where
  /-- `N`: the rows, `x.numel() / D` (the product of the leading dimensions). -/
  N : Nat
  /-- `D`: `x.shape[-1]`, the row length; the constexpr `D` and the argument `stride_z_row`. -/
  D : Nat
  /-- `B`: the elements per program (the host passes `4096`; fla's autotune list holds `512` to
  `8192`). -/
  B : Nat
  /-- `stride_x_row`: the row stride of `x`, in elements (`prepare_input`). -/
  sx : Nat
  /-- `stride_y_row`: the row stride of `y`, in elements. -/
  sy : Nat
  /-- `x`: `bfloat16`; `x[r, c]` is element `r·sx + c`. -/
  x : Buf .bf16
  /-- `y`: `bfloat16`; `y[r, c]` is element `r·sy + c`. -/
  y : Buf .bf16
  /-- `z`: the output, `bfloat16`, contiguous; `z[r, c]` is element `r·D + c`. -/
  z : Buf .bf16

/-- `T = x.numel()`, the flat element count. -/
def Inst.T (d : Inst) : Nat := d.N * d.D

/-- The elements of an input with row stride `s` the launch reads: `r·s + c`, `r < N`, `c < D`. -/
def inSet (d : Inst) (s : Nat) : Set Nat := {k | ∃ r c, r < d.N ∧ c < d.D ∧ k = r * s + c}

/-- The elements of `z` the launch writes: the first `N·D`. -/
def outSet (d : Inst) : Set Nat := {k | k < d.N * d.D}

/-- The instances covered: every row length `D ≥ 1` (below `2^31`), every power-of-two `B` of a
block Triton accepts (the host's `4096` and every `B` of fla's autotune list), every number of rows
whose grid CUDA accepts, any row strides of the inputs (contiguous rows, the halves of one
projection, broadcast rows), and an output that shares no byte with the inputs' elements read. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `D ≥ 1`: a tensor's last dimension (the kernel divides by it). -/
  d_pos : 0 < d.D
  /-- `D < 2^31`: the constexpr `D` and the argument `stride_z_row = D` are `int32`-sized (real
  row lengths are below `2^17`). -/
  d_lt : d.D < 2 ^ 31
  /-- `B` is a power of two (Triton requires it of `tl.arange`). -/
  b_pow2 : ∃ e, d.B = 2 ^ e
  /-- `B` is at most `2^20` elements (Triton's largest block). -/
  b_le : d.B ≤ 2 ^ 20
  /-- The grid `cdiv(T, B)` has fewer than `2^31` programs (CUDA's limit on axis 0; a larger grid
  is rejected at launch). With `B ≤ 2^20` it also gives `T < 2^51`. -/
  grid_lt : (d.N * d.D + d.B - 1) / d.B < 2 ^ 31
  /-- `stride_x_row` is an integer argument Triton accepts as `int32` or `int64` (below `2^63`). -/
  sx_lt : d.sx < 2 ^ 63
  /-- `stride_y_row` likewise. -/
  sy_lt : d.sy < 2 ^ 63
  /-- Row `r` of `x` is in the buffer. -/
  x_len : ∀ r, r < d.N → r * d.sx + d.D ≤ d.x.len
  /-- Row `r` of `y` is in the buffer. -/
  y_len : ∀ r, r < d.N → r * d.sy + d.D ≤ d.y.len
  /-- `z` holds the `N·D` elements the launch writes. -/
  z_len : d.N * d.D ≤ d.z.len
  /-- The elements of `z` the launch writes share no byte with the elements of `x` it reads (the
  host allocates `z`). -/
  z_x : Disjoint (d.z.elems (outSet d)) (d.x.elems (inSet d d.sx))
  /-- The elements of `z` the launch writes share no byte with the elements of `y` it reads. -/
  z_y : Disjoint (d.z.elems (outSet d)) (d.y.elems (inSet d d.sy))

/-- The model on an instance: the buffers' first elements, the integer arguments as Triton passes
them (`argOf`: `int32` below `2^31`, `int64` from `2^31`), `stride_z_row = D` (the host's), and the
constexprs `D`, `B`. -/
def model (d : Inst) : Kernel :=
  swiglu_fwd_kernel (d.x.ptr 0) (d.y.ptr 0) (d.z.ptr 0) (argOf d.sx) (argOf d.sy) (argOf d.D)
    (argOf d.T) d.D d.B

/-- The host's launch (source lines 112–123): the grid `(triton.cdiv(T, B),)`, `(T + B − 1) / B`.
The source's only block shape is `[B]` (`tl.arange(0, B)`, `offs`, `mask`, `row`, `col`, the
pointers, the loads and every value computed from them). The launch passes `num_warps` (performance
only) and no other option, so `enable_fp_fusion` is Triton's default `true`, `debug` is false,
`num_ctas = 1`, no `launch_pdl`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.N * d.D + d.B - 1) / d.B) [[d.B]] (fusion := true)

/-! ## The specification -/

/-- The exact output `a · sigmoid(a) · b = a · b / (1 + e^-a)`. -/
noncomputable def zRef (a b : ℝ) : ℝ := a * b / (1 + Real.exp (-a))

/-- The tolerance of an output whose exact value is `v`: `(2^-8 + 2^-13)·|v| + 2^-133` (the
`bfloat16` rounding, `2^-8` relative, `2^-134` below its normal range, and the `float32` steps
before it; README, Tolerances). -/
noncomputable def zTol (v : ℝ) : ℝ := (2 ^ (-8 : ℤ) + 2 ^ (-13 : ℤ)) * |v| + 2 ^ (-133 : ℤ)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); every element the launch reads is a finite `bfloat16`, and every exact result is at most
`2^127` in magnitude (beyond it the `float32` product or the `bfloat16` rounding may overflow to
`±∞`; README, Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.y.Valid init ∧ d.z.Valid init ∧
    ∀ r, r < d.N → ∀ c, c < d.D → ∃ a b : BF16,
      init.read .bf16 (d.x.ptr (r * d.sx + c)).addr = some a ∧
      init.read .bf16 (d.y.ptr (r * d.sy + c)).addr = some b ∧
      a.isFinite = true ∧ b.isFinite = true ∧ |zRef a.toReal b.toReal| ≤ 2 ^ 127

/-- An output `o` for the inputs `a`, `b` is right: a finite `bfloat16`, zero when `a ≤ -89`
(where `tl.exp(-a)` overflows), and within `zTol` of `zRef a b` otherwise. -/
def ZOk (a b o : BF16) : Prop :=
  o.isFinite = true ∧ (a.toReal ≤ -89 → o.toReal = 0) ∧
    (-89 < a.toReal → |o.toReal - zRef a.toReal b.toReal| ≤ zTol (zRef a.toReal b.toReal))

/-- What the launch leaves: for each row `r < N` and column `c < D`, with `a = x[r, c]` and
`b = y[r, c]` the initial inputs, `z[r, c]` satisfies `ZOk a b`; every byte outside the first `N·D`
elements of `z` is unchanged (so the inputs are unchanged). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.N → ∀ c, c < d.D → ∃ a b o : BF16,
      init.read .bf16 (d.x.ptr (r * d.sx + c)).addr = some a ∧
      init.read .bf16 (d.y.ptr (r * d.sy + c)).addr = some b ∧
      fin.read .bf16 (d.z.ptr (r * d.D + c)).addr = some o ∧ ZOk a b o) ∧
    ∀ w, w ∉ d.z.elems (outSet d) → fin w = init w

/-! ## The claim -/

/-- **The claim**: every covered launch of `swiglu_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- A launch of the host on `N` rows of `D` elements with `B = 2^e`. Contiguous (`split = false`):
`x`, `y`, `z` of `N·D` elements at the byte addresses `2^44`, `2^45`, `2^46`, row strides `D`.
Split (`split = true`): `x` and `y` the two halves of one `[N, 2·D]` projection
(`gate, y = proj.chunk(2, dim=-1)`), row stride `2·D`, `y` starting `D` elements after `x`; `z`
contiguous. -/
def mk (N D : Nat) (split : Bool := false) (e : Nat := 12) : Inst where
  N := N
  D := D
  B := 2 ^ e
  sx := if split then 2 * D else D
  sy := if split then 2 * D else D
  x := if split then ⟨2 ^ 44, N * (2 * D)⟩ else ⟨2 ^ 44, N * D⟩
  y := if split then ⟨2 ^ 44 + D * 2, N * (2 * D)⟩ else ⟨2 ^ 45, N * D⟩
  z := ⟨2 ^ 46, N * D⟩

theorem bytes_disjoint {d₁ d₂ : DType} (b : Buf d₁) (c : Buf d₂)
    (h : b.base + b.len * d₁.bytes ≤ c.base ∨ c.base + c.len * d₂.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro w hw hw'
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hw hw'
  omega

theorem rows_fit {N s D len : Nat} (hs : D ≤ s) (h : N * s ≤ len) :
    ∀ r, r < N → r * s + D ≤ len := by
  intro r hr
  have : (r + 1) * s ≤ N * s := Nat.mul_le_mul_right _ hr
  rw [Nat.succ_mul] at this
  omega

theorem inSet_lt {d : Inst} {s len : Nat} (hs : d.D ≤ s) (h : d.N * s ≤ len) :
    ∀ j ∈ inSet d s, j < len := by
  rintro j ⟨r, c, hr, hc, rfl⟩
  have : (r + 1) * s ≤ d.N * s := Nat.mul_le_mul_right _ hr
  rw [Nat.succ_mul] at this
  omega

theorem mk_ok (N D : Nat) (split : Bool) (e : Nat) (hD : 0 < D) (hD' : D < 2 ^ 31)
    (hND : N * D ≤ 2 ^ 39) (he : 9 ≤ e) (he' : e ≤ 20) : (mk N D split e).ok := by
  have hB9 : 2 ^ 9 ≤ 2 ^ e := Nat.pow_le_pow_right (by norm_num) he
  have hB20 : 2 ^ e ≤ 2 ^ 20 := Nat.pow_le_pow_right (by norm_num) he'
  have hgrid : (N * D + 2 ^ e - 1) / 2 ^ e < 2 ^ 31 := by
    rw [Nat.div_lt_iff_lt_mul (by positivity)]
    have : 2 ^ 31 * 2 ^ 9 ≤ 2 ^ 31 * 2 ^ e := Nat.mul_le_mul_left _ hB9
    omega
  have h2 : N * (2 * D) = 2 * (N * D) := by ring
  have hbf : DType.bytes DType.bf16 = 2 := rfl
  cases split <;>
  refine ⟨hD, hD', ⟨e, rfl⟩, hB20, hgrid, ?_, ?_, rows_fit ?_ ?_, rows_fit ?_ ?_, le_refl _, ?_, ?_⟩ <;>
  simp only [mk, Bool.false_eq_true, ↓reduceIte] <;>
  try omega
  all_goals
    refine Buf.elems_disjoint_of _ _ (bytes_disjoint _ _ ?_) (fun j hj => hj)
      (inSet_lt (by dsimp only; omega) (le_refl _))
    right
    simp only [hbf]
    have h2' : ((N : Int) * (2 * (D : Int))) = 2 * ((N * D : Nat) : Int) := by push_cast; ring
    push_cast at h2' ⊢
    omega

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

/-- The source's first test case (`x`, `y` of shape `(1, 1)`) on the all-zero memory: both inputs
are `+0`, and so is the exact result. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 1, zeroMem, mk_ok 1 1 false 12 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num), ?_, ?_, ?_, ?_⟩
  all_goals try
    exact ⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [zeroMem]⟩
  intro r hr c hc
  have hr0 : r = 0 := by simp [mk] at hr; omega
  have hc0 : c = 0 := by simp [mk] at hc; omega
  subst hr0; subst hc0
  refine ⟨FloatV.ofPat 0, FloatV.ofPat 0, zeroMem_read16 ?_, zeroMem_read16 ?_, by decide,
    by decide, ?_⟩
  · unfold Mem.Aligned; simp [Buf.ptr_addr, mk, DType.bytes]
  · unfold Mem.Aligned; simp [Buf.ptr_addr, mk, DType.bytes]
  · have hz : (FloatV.ofPat 0 : BF16).toReal = 0 := by
      show ((Format.bfloat16.toRat 0 : ℚ) : ℝ) = 0
      have : Format.bfloat16.toRat 0 = 0 := by decide +kernel
      rw [this]; simp
    rw [hz]
    simp only [zRef, zero_mul, zero_div, abs_zero]
    positivity

/-! ## Coverage: the launches SETUP.md lists

The source's test (`(1, 1)`, `(7, 33)`, `(4, 1024, 512)`, `(2, 2048, 2752)` contiguous, and both
halves of a `(4, 1024, 1376)` projection), fla's `GatedMLP` at hidden sizes 2048 and 4096 (`D` 5632
and 11008) on training batches and on one decoding token, every `B` of fla's autotune list, and a
batch of `2^32` elements (`T` an `int64`). -/

def setup_test_1_1 : Inst := mk 1 1
theorem setup_test_1_1_ok : setup_test_1_1.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_test_7_33 : Inst := mk 7 33
theorem setup_test_7_33_ok : setup_test_7_33.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_test_4_1024_512 : Inst := mk 4096 512
theorem setup_test_4_1024_512_ok : setup_test_4_1024_512.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_test_2_2048_2752 : Inst := mk 4096 2752
theorem setup_test_2_2048_2752_ok : setup_test_2_2048_2752.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_test_chunk_4_1024_688 : Inst := mk 4096 688 true
theorem setup_test_chunk_4_1024_688_ok : setup_test_chunk_4_1024_688.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `GatedMLP(hidden_size=2048)` (`D = 5632`) on 8 × 2048 tokens. -/
def setup_mlp2048_train : Inst := mk 16384 5632
theorem setup_mlp2048_train_ok : setup_mlp2048_train.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `GatedMLP(hidden_size=2048)` on one decoding token. -/
def setup_mlp2048_decode : Inst := mk 1 5632
theorem setup_mlp2048_decode_ok : setup_mlp2048_decode.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `GatedMLP(hidden_size=4096)` (`D = 11008`) on 4 × 4096 tokens. -/
def setup_mlp4096_train : Inst := mk 16384 11008
theorem setup_mlp4096_train_ok : setup_mlp4096_train.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The autotune list's `B = 512`, `1024`, `2048`, `8192` (`4096` is the host's default above),
on the source's `(2, 2048, 2752)` test. -/
def setup_b512 : Inst := mk 4096 2752 false 9
theorem setup_b512_ok : setup_b512.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_b1024 : Inst := mk 4096 2752 false 10
theorem setup_b1024_ok : setup_b1024.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_b2048 : Inst := mk 4096 2752 false 11
theorem setup_b2048_ok : setup_b2048.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_b8192 : Inst := mk 4096 2752 false 13
theorem setup_b8192_ok : setup_b8192.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `2^19` tokens of `D = 8192`: `T = 2^32`, an `int64` argument. -/
def setup_t64 : Inst := mk (2 ^ 19) 8192
theorem setup_t64_ok : setup_t64.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.SwigluFwd
