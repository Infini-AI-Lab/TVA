import TVA

/-!
# `logsigmoid_fwd_bf16`: log-sigmoid with a temperature, forward (fla)

The kernel is `logsigmoid_fwd_kernel` of `source/logsigmoid.py` (lines 92–103; it reads the
module-level names `triton` and `tl`, bound by the imports of lines 49–50, and calls the
`@triton.jit` helpers `exp` and `log` of lines 53–56). Its host `logsigmoid_fwd` (lines 106–121)
launches it on the grid `(triton.cdiv(T, 1024),)` with `B = 1024` and `num_warps = 4`, on a
`bfloat16` `x` of `T = x.numel()` elements in rows of `D = x.shape[-1]` (unit stride along the
row, one row stride `stride_x_row` for the flattened rows, `prepare_input`) and a fresh contiguous
`bfloat16` `y` of the same shape; `SETUP.md` lists the launches. Element `k < T` is column `k % D` of
row `k // D`.

What is claimed: every covered launch ends normally, and for each `k < T`, with `v` the initial
`bfloat16` input at row `k // D`, column `k % D` and `t` the host's double `temperature`: `y[k]`
is a finite `bfloat16` within `yTol v t = (2^-8 + 2^-19)·|lsRef v / t| + 2^-23 / |t| + 2^-133` of
`lsRef v / t`, where `lsRef v = -log(1 + e^-v)` is the exact log-sigmoid; every byte outside the
first `T` elements of `y` is unchanged.

The conditions: `B` a power of two at most `2^20`; fewer than `2^31` programs; a row stride below
`2^63`; the buffers hold the elements the launch reads and writes, and the written elements of `y`
share no byte with the read elements of `x`; `temperature` a finite double with
`2^-126 ≤ |t| ≤ 2^127`; every input read finite, with `|v| + 1 ≤ 2^125·|t|`.

The tolerance rests on the device facts for `tl.exp`, `tl.log` and `/` on `float32`
(`Device.ExpF32`, `Device.LogF32`, `Device.DivFull`, and their totality facts); README,
Tolerances derives it.
-/

namespace Claims.LogsigmoidFwd

open TVA

/-! ## The model -/

/-- The `float32` constant Triton builds for the Python number `1.0` as an operand of `+` beside a
`float32` tensor (`float64` pattern `0x3FF0000000000000`, `litF32`). -/
def oneF32 : F32 := FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)

/-- The `float32` tensor the Python number `0.` becomes as an argument of `tl.minimum` (a
`float` zero is a `float32` tensor of its own dtype, `+0`; `litF32` of the pattern `0`). -/
def zeroF32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- A runtime integer argument used in an `int64` operation: an `int32` one is promoted to `int64`
by the operation (the sign extension, `tl.convert .i32 .i64`); an `int64` one is used as it is. -/
def arg64 : Sum I32 I64 → Prog I64
  | .inl t => do pure (← tl.convert .i32 .i64 (Tensor.scalar t)).item
  | .inr t => pure t

/-- `logsigmoid_fwd_kernel` (source lines 92–103). `x` and `y` are `*bf16` (the host's `bfloat16`
tensors). `stride_x_row`, `stride_y_row` and `T` are unannotated integer arguments: a runtime
`int32` below `2^31` and a runtime `int64` from `2^31` (`Sum.inl`, `Sum.inr`; `model` picks by the
value). `T` is in `do_not_specialize`, so it is never the compile-time constant `1`; a stride equal
to `1` is the constant `1`, which its only use, `row * stride` with the `int64` tensor `row`, builds
in `int64`, the value `arg64` gives the `int32` `1`, so the model is the same. `temperature` is a
Python `float` argument: a runtime `float32` scalar, `FloatV.hostF32` of the host's double
(`model`). `D` and `B` are `tl.constexpr`s, Lean parameters. -/
def logsigmoid_fwd_kernel (x y : Ptr .bf16) (stride_x_row stride_y_row : Sum I32 I64)
    (temperature : F32) (T : Sum I32 I64) (D B : Nat) : Kernel where
  body ctx := do
    -- line 93: `tl.program_id(0).to(tl.int64)`: axis 0 of the grid, an `int32` scalar,
    -- sign-extended to `int64`
    let i_n : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 94: `i_n * B + tl.arange(0, B)`: the constexpr `B` built in `int64` (the dtype of the
    -- scalar `i_n`), the `int64` product; the `int32` block `tl.arange(0, B)` sign-extended to
    -- `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [B] I64 ← tl.convert .i32 .i64 (tl.arange 0 B)
    let offs : Tensor [B] I64 := i_n * (IntV.ofNat B : I64) + o_a
    -- line 95: `offs < T`: the scalar `T` promoted to `int64` (`arg64`), the signed comparison
    let T64 ← arg64 T
    let mask : Tensor [B] Bool := offs.ltS T64
    -- lines 96-97: `//` and `%` of the `int64` block by the constexpr `D` built in `int64`
    -- (truncating, in every lane, the disabled ones included)
    let row ← tl.idiv offs (tl.full [B] (IntV.ofNat D : I64))
    let col ← tl.irem offs (tl.full [B] (IntV.ofNat D : I64))
    -- line 98: `x + row * stride_x_row + col`, evaluated left to right: the `int64` product of
    -- `row` and the stride (`arg64`), added to the `*bf16` pointer, then `col`. The masked load
    -- with `other=0.`: the Python `0.` is a `float32` tensor (`+0`), converted to the pointer's
    -- dtype, `+0` in `bfloat16`; then `.to(tl.float32)`, the exact widening (`tl.cvt`)
    let sx ← arg64 stride_x_row
    let b_x16 : Tensor [B] BF16 ←
      tl.load (x + row * sx + col) (some mask) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_x : Tensor [B] F32 ← tl.cvt b_x16
    -- line 99: `tl.minimum(0., b_x)`: the Python `0.` a `float32` tensor (`zeroF32`), broadcast;
    -- the float minimum (`FloatV.MaxRel`)
    let b_m ← tl.minimum (bc zeroF32) b_x
    -- line 100: `1. + exp(-tl.abs(b_x))`: `tl.abs`, the unary `-` (`tl.neg`); the helper `exp`
    -- (line 54) inlined, `tl.exp(x.to(tl.float32))`, whose `.to` is no conversion (`float32`);
    -- the Python `1.` built in `float32` (`oneF32`) and the IEEE `+` (its operand is
    -- `tl.exp`'s result, not a product)
    let b_a ← tl.abs b_x
    let b_na ← tl.neg b_a
    let b_e ← tl.exp b_na
    let b_z ← tl.add (bc oneF32) b_e
    -- line 101: `(b_m - log(b_z)) / temperature`: the helper `log` (line 56) inlined,
    -- `tl.log(x.to(tl.float32))`, no conversion; the IEEE `-` (no product feeds it); the
    -- platform's division by the runtime `float32` scalar, broadcast (`tl.div`, `div.full.f32`;
    -- no addition consumes the quotient)
    let b_l ← tl.log b_z
    let b_s ← tl.sub b_m b_l
    let b_y ← tl.div b_s (bc temperature)
    -- line 102: `y + row * stride_y_row + col`, as line 98; `b_y.to(y.dtype.element_ty)` rounds to
    -- `bfloat16` (to nearest even, `tl.cvt`); the masked store
    let sy ← arg64 stride_y_row
    let b_y16 : Tensor [B] BF16 ← tl.cvt b_y
    tl.store (y + row * sy + col) b_y16 (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `logsigmoid_fwd_kernel`. -/
structure Inst where
  /-- The number of rows of `x` flattened to `[N, D]` (the product of its leading dimensions). -/
  N : Nat
  /-- `D`: `x.shape[-1]`, the row length (a constexpr, and the runtime `stride_y_row`). -/
  D : Nat
  /-- `stride_x_row`: the row stride of `x` in elements (`prepare_input`: `D` for a contiguous
  tensor, the stride of the flattened rows for an inner-contiguous view). -/
  sx : Nat
  /-- `B`: the elements per program (the host passes `1024`; the autotune list of fla's source,
  `512` to `8192`, are powers of two). -/
  B : Nat
  /-- `temperature`: the host's Python `float` (a double); the kernel gets `FloatV.hostF32` of it. -/
  temperature : F64
  /-- `x`: the input, `bfloat16`; row `i`, column `j` is element `i·sx + j`. -/
  x : Buf .bf16
  /-- `y`: the output, `bfloat16`, contiguous; element `k` is row `k / D`, column `k % D`. -/
  y : Buf .bf16

/-- `T = x.numel()`, the number of elements: `N·D`. -/
def Inst.T (d : Inst) : Nat := d.N * d.D

/-- The element of `x` that element `k` of `y` is computed from: row `k / D`, column `k % D`. -/
def xIdx (d : Inst) (k : Nat) : Nat := k / d.D * d.sx + k % d.D

/-- The elements of `x` the launch reads. -/
def xSet (d : Inst) : Set Nat := {j | ∃ k, k < d.T ∧ j = xIdx d k}

/-- The elements of `y` the launch writes: the first `T`. -/
def ySet (d : Inst) : Set Nat := {k | k < d.T}

/-- The instances covered: every block size Triton accepts, every grid CUDA accepts, any row
stride Triton passes as an integer argument, buffers holding the elements the launch touches, an
output that shares no byte with the input elements read, and a temperature whose `float32` is a
normal number. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `B` is a power of two (Triton requires it of `tl.arange`). -/
  block_pow2 : ∃ e, d.B = 2 ^ e
  /-- `B` is at most `2^20` elements (Triton's largest block). -/
  block_le : d.B ≤ 2 ^ 20
  /-- The grid `cdiv(T, B)` has fewer than `2^31` programs (CUDA's limit on axis 0). With
  `B ≤ 2^20` it also gives `T < 2^51`, inside `int64`. -/
  grid_lt : (d.T + d.B - 1) / d.B < 2 ^ 31
  /-- The row stride is an integer argument below `2^63` (an `int64`). -/
  sx_lt : d.sx < 2 ^ 63
  /-- Row `i` of `x` is in the buffer (any row stride, so contiguous rows, the halves of a
  `chunk(2, dim=-1)`, and rows read more than once). -/
  x_len : ∀ i, i < d.N → i * d.sx + d.D ≤ d.x.len
  /-- `y` holds the `T` elements the launch writes. -/
  y_len : d.T ≤ d.y.len
  /-- The elements of `y` the launch writes share no byte with the elements of `x` it reads (the
  host allocates `y` fresh). -/
  y_x : Disjoint (d.y.elems (ySet d)) (d.x.elems (xSet d))
  /-- `temperature` is a finite double. -/
  t_fin : d.temperature.isFinite = true
  /-- `|temperature| ≥ 2^-126`: its `float32` is a normal number, within `2^-24` (relative) of it.
  Below, the `float32` rounds it with a larger relative error, or to `0`. -/
  t_lo : (2 : ℝ) ^ (-126 : ℤ) ≤ |d.temperature.toReal|
  /-- `|temperature| ≤ 2^127`: its `float32` is finite. -/
  t_hi : |d.temperature.toReal| ≤ 2 ^ 127

/-- A runtime integer argument of value `n` as Triton passes it: an `int32` below `2^31`, an
`int64` from `2^31`. -/
def intArg (n : Nat) : Sum I32 I64 :=
  if n < 2 ^ 31 then Sum.inl (IntV.ofNat n) else Sum.inr (IntV.ofNat n)

/-- The model on an instance: the buffers' first elements, the strides `stride_x_row` and
`stride_y_row = D` and `T` as the runtime integers Triton passes, `temperature` as the `float32`
the device gets (`FloatV.hostF32` of the double), and the constexprs `D`, `B`. -/
def model (d : Inst) : Kernel :=
  logsigmoid_fwd_kernel (d.x.ptr 0) (d.y.ptr 0) (intArg d.sx) (intArg d.D)
    (FloatV.hostF32 d.temperature) (intArg d.T) d.D d.B

/-- The host's launch (source lines 109–120): the grid `(triton.cdiv(T, B),)`, `(T + B − 1) / B`.
The source's only block shape is `[B]` (`tl.arange`, `offs`, `mask`, `row`, `col`, the pointers
and every value). The launch passes `num_warps = 4` and no other option, so `enable_fp_fusion` is
Triton's default `true`, `debug` is false, `num_ctas = 1`, no `launch_pdl`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.T + d.B - 1) / d.B) [[d.B]] (fusion := true)

/-- The inputs covered: the two buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); every element the launch reads is a finite `bfloat16` `v` with `|v| + 1 ≤ 2^125·|t|`
(the quotient by the temperature stays below `2^126`, where `Device.DivFull` holds; README,
Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.y.Valid init ∧
    ∀ k, k < d.T → ∃ v : BF16, init.read .bf16 (d.x.ptr (xIdx d k)).addr = some v ∧
      v.isFinite = true ∧ |v.toReal| + 1 ≤ 2 ^ 125 * |d.temperature.toReal|

/-! ## The specification -/

/-- The exact log-sigmoid, `log(1 / (1 + e^-v)) = -log(1 + e^-v)`. -/
noncomputable def lsRef (v : ℝ) : ℝ := -Real.log (1 + Real.exp (-v))

/-- The tolerance of `y[k]` at input `v` and temperature `t`:
`(2^-8 + 2^-19)·|lsRef v / t| + 2^-23 / |t| + 2^-133` (README, Tolerances). The absolute term
`2^-23 / |t|` is the kernel's: `log(1 + e^-|v|)` computed as `tl.log` of the rounded `1 + e^-|v|`
is within about `2^-24` of the exact value, not within a relative bound, for large `v`. -/
noncomputable def yTol (v t : ℝ) : ℝ :=
  (2 ^ (-8 : ℤ) + 2 ^ (-19 : ℤ)) * |lsRef v / t| + 2 ^ (-23 : ℤ) / |t| + 2 ^ (-133 : ℤ)

/-- What the launch leaves: for each `k < T`, with `v` the initial element `xIdx d k` of `x` and
`t` the host's double `temperature`, element `k` of `y` is a finite `bfloat16` within
`yTol v t` of `lsRef v / t`; every byte outside the first `T` elements of `y` is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ k, k < d.T → ∃ v : BF16, init.read .bf16 (d.x.ptr (xIdx d k)).addr = some v ∧
      ∃ c : BF16, fin.read .bf16 (d.y.ptr k).addr = some c ∧ c.isFinite = true ∧
        |c.toReal - lsRef v.toReal / d.temperature.toReal| ≤
          yTol v.toReal d.temperature.toReal) ∧
    ∀ z, z ∉ d.y.elems (ySet d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `logsigmoid_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Instances

`mk N D split t`: the host's tensors at fixed, 16-byte-aligned, disjoint addresses, with
`B = 1024` and the double `t` (by its pattern). Contiguous (`split = false`): `x` of `N·D`
elements, row stride `D`. Split (`split = true`): `x` the first half of a `[N, 2·D]` tensor
(`x.chunk(2, dim=-1)[0]`), row stride `2·D`. `y` holds `N·D` elements. -/

def mk (N D : Nat) (split : Bool) (t : F64) : Inst where
  N := N
  D := D
  sx := if split then 2 * D else D
  B := 1024
  temperature := t
  x := if split then ⟨2 ^ 34, (N - 1) * (2 * D) + D⟩ else ⟨2 ^ 34, N * D⟩
  y := ⟨2 ^ 36, N * D⟩

/-- The double `1.0`, pattern `0x3FF0000000000000`. -/
def t1 : F64 := FloatV.ofPat 0x3FF0000000000000
/-- The double `0.5`, pattern `0x3FE0000000000000`. -/
def t05 : F64 := FloatV.ofPat 0x3FE0000000000000
/-- The double `2.0`, pattern `0x4000000000000000`. -/
def t2 : F64 := FloatV.ofPat 0x4000000000000000
/-- The double `16.0` (GLA's `gate_logit_normalizer`), pattern `0x4030000000000000`. -/
def t16 : F64 := FloatV.ofPat 0x4030000000000000

theorem bytes_disj {d₁ d₂ : DType} (x : Buf d₁) (y : Buf d₂)
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem rows_fit {N s D len : Nat} (h : (N - 1) * s + D ≤ len) :
    ∀ i, i < N → i * s + D ≤ len := by
  intro i hi
  have : i * s ≤ (N - 1) * s := Nat.mul_le_mul_right _ (by omega)
  omega

/-- A temperature whose double is finite with `2^-126 ≤ |t| ≤ 2^127`. -/
def TOk (t : F64) : Prop :=
  t.isFinite = true ∧ (2 : ℝ) ^ (-126 : ℤ) ≤ |t.toReal| ∧ |t.toReal| ≤ 2 ^ 127

theorem tok_of_rat {t : F64} (hf : t.isFinite = true) (hlo : (2 : ℚ) ^ (-126 : ℤ) ≤ t.toRat)
    (hhi : t.toRat ≤ 2 ^ 127) : TOk t := by
  have h0 : (0 : ℚ) ≤ t.toRat := le_trans (by positivity) hlo
  have hlo' : ((2 : ℚ) ^ (-126 : ℤ) : ℝ) ≤ (t.toRat : ℝ) := by exact_mod_cast hlo
  have hhi' : (t.toRat : ℝ) ≤ ((2 : ℚ) ^ 127 : ℝ) := by exact_mod_cast hhi
  have h0' : (0 : ℝ) ≤ (t.toRat : ℝ) := by exact_mod_cast h0
  refine ⟨hf, ?_, ?_⟩ <;> simp only [FloatV.toReal, abs_of_nonneg h0'] <;> push_cast at hlo' hhi' ⊢ <;>
    assumption

theorem tok_t1 : TOk t1 := tok_of_rat (by decide) (by decide +kernel) (by decide +kernel)
theorem tok_t05 : TOk t05 := tok_of_rat (by decide) (by decide +kernel) (by decide +kernel)
theorem tok_t2 : TOk t2 := tok_of_rat (by decide) (by decide +kernel) (by decide +kernel)
theorem tok_t16 : TOk t16 := tok_of_rat (by decide) (by decide +kernel) (by decide +kernel)

/-- Every element of `x` the launch reads is in the buffer. -/
theorem xIdx_lt {d : Inst} (hx : ∀ i, i < d.N → i * d.sx + d.D ≤ d.x.len) {k : Nat}
    (hk : k < d.T) : xIdx d k < d.x.len := by
  have hD : 0 < d.D := by
    rcases Nat.eq_zero_or_pos d.D with h | h
    · simp [Inst.T, h] at hk
    · exact h
  have hr : k / d.D < d.N := (Nat.div_lt_iff_lt_mul hD).mpr hk
  have hc : k % d.D < d.D := Nat.mod_lt _ hD
  have := hx _ hr
  unfold xIdx
  omega

theorem mk_ok (N D : Nat) (split : Bool) (t : F64) (ht : TOk t) (hN : 1 ≤ N)
    (hND : N * D ≤ 2 ^ 31) : (mk N D split t).ok := by
  have hND2 : (N - 1) * (2 * D) + D ≤ 2 * (N * D) := by
    have : (N - 1) * (2 * D) + 2 * D = 2 * (N * D) := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hN]; ring
    omega
  have hND1 : (N - 1) * D + D = N * D := by
    rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hN]
  have bf : DType.bytes DType.bf16 = 2 := rfl
  obtain ⟨hf, hlo, hhi⟩ := ht
  have hdisj : Disjoint (mk N D split t).y.bytes (mk N D split t).x.bytes := by
    apply bytes_disj
    cases split <;> simp only [mk, bf, Bool.false_eq_true, ↓reduceIte] <;> omega
  refine ⟨⟨10, rfl⟩, by simp [mk], ?_, ?_, ?_, ?_, ?_, hf, hlo, hhi⟩
  · simp only [mk, Inst.T]
    omega
  · cases split <;> simp only [mk, Bool.false_eq_true, ↓reduceIte] <;> omega
  · cases split <;> simp only [mk, Bool.false_eq_true, ↓reduceIte]
    · exact rows_fit (by omega)
    · exact rows_fit (by omega)
  · simp only [mk, Inst.T]; exact le_refl _
  · refine hdisj.mono (Buf.elems_sub_bytes _ fun j hj => ?_) (Buf.elems_sub_bytes _ fun j hj => ?_)
    · simp only [ySet, Set.mem_ofPred_eq, mk, Inst.T] at hj ⊢; exact hj
    · obtain ⟨k, hk, rfl⟩ := hj
      refine xIdx_lt (fun i hi => ?_) hk
      cases split <;> simp only [mk, Bool.false_eq_true, ↓reduceIte] at hi ⊢
      · exact rows_fit (by omega) i hi
      · exact rows_fit (by omega) i hi

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem readBytes_zeroMem : ∀ (n : Nat) (a : Int),
    zeroMem.readBytes a n = some (List.replicate n 0) := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    show (match zeroMem a, zeroMem.readBytes (a + 1) n with
      | some b, some bs => some (b :: bs)
      | _, _ => none) = _
    rw [ih (a + 1)]
    rfl

theorem zeroMem_read {d : DType} {a : Int} (h : Mem.Aligned d a) :
    zeroMem.read d a = some (DType.ofBits d 0) := by
  simp only [Mem.read, h, ↓reduceIte, readBytes_zeroMem, Option.map_some]
  have hlv : ∀ n : Nat, Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Bytes.leValue, ih]; norm_num
  rw [hlv]

theorem zero_bf16 : (DType.ofBits .bf16 0 : BF16).isFinite = true ∧
    (DType.ofBits .bf16 0 : BF16).toReal = 0 := by
  refine ⟨by decide, ?_⟩
  have : (DType.ofBits .bf16 0 : BF16).toRat = 0 := by decide +kernel
  simp [FloatV.toReal, this]

/-- The source's first test case (`x` of shape `(1, 1, 64)`, contiguous, `temperature = 1.0`) on
the all-zero memory: every input is `0`. -/
theorem claim_satisfiable : claim.Satisfiable := by
  have bf : DType.bytes DType.bf16 = 2 := rfl
  refine ⟨mk 1 64 false t1, zeroMem, mk_ok 1 64 false t1 tok_t1 (by norm_num) (by norm_num), ?_⟩
  refine ⟨⟨by simp [mk], by simp [mk, bf], by simp [mk, bf], fun _ _ => by simp [zeroMem]⟩,
    ⟨by simp [mk], by simp [mk, bf], by simp [mk, bf], fun _ _ => by simp [zeroMem]⟩, ?_⟩
  intro k hk
  simp only [mk, Inst.T] at hk
  refine ⟨_, zeroMem_read ⟨?_, ?_, ?_⟩, zero_bf16.1, ?_⟩
  · simp only [mk, Buf.ptr_addr, bf, Bool.false_eq_true, ↓reduceIte]; positivity
  · simp only [mk, Buf.ptr_addr, bf, Bool.false_eq_true, ↓reduceIte, xIdx]; push_cast; omega
  · simp only [mk, Buf.ptr_addr, bf, Bool.false_eq_true, ↓reduceIte, xIdx]; push_cast; omega
  · rw [zero_bf16.2, abs_zero, zero_add]
    have h1 : t1.toRat = 1 := by decide +kernel
    have h1' : (mk 1 64 false t1).temperature.toReal = 1 := by
      simp only [mk, FloatV.toReal, h1, Rat.cast_one]
    rw [h1', abs_one, mul_one]
    norm_num

/-! ## The setup's launches

fla's test cases (`tests/modules/test_activation.py:104-116`: shapes `(1, 1, 64)`, `(2, 500, 128)`,
`(2, 512, 128)`, `(3, 2048, 1200)` with temperatures `1.0`, `0.5`, `0.5`, `2.0`, contiguous and as
the first half of `chunk(2, dim=-1)`, and `(63,)`), then SETUP.md's model-sized launches: GLA's
gate logits (`4 × 2048` tokens of key dimension `1024`, temperature `16.0`) and its output gate
(`4 × 2048` tokens of value dimension `2048`, temperature `1.0`), and a long sequence (`65536`
tokens of `4096`). -/

def setup_1_1_64 : Inst := mk 1 64 false t1
theorem setup_1_1_64_ok : setup_1_1_64.ok := mk_ok _ _ _ _ tok_t1 (by norm_num) (by norm_num)
def setup_2_500_128 : Inst := mk 1000 128 false t05
theorem setup_2_500_128_ok : setup_2_500_128.ok := mk_ok _ _ _ _ tok_t05 (by norm_num) (by norm_num)
def setup_2_512_128 : Inst := mk 1024 128 false t05
theorem setup_2_512_128_ok : setup_2_512_128.ok := mk_ok _ _ _ _ tok_t05 (by norm_num) (by norm_num)
def setup_3_2048_1200 : Inst := mk 6144 1200 false t2
theorem setup_3_2048_1200_ok : setup_3_2048_1200.ok :=
  mk_ok _ _ _ _ tok_t2 (by norm_num) (by norm_num)
def setup_2_500_128_split : Inst := mk 1000 128 true t05
theorem setup_2_500_128_split_ok : setup_2_500_128_split.ok :=
  mk_ok _ _ _ _ tok_t05 (by norm_num) (by norm_num)
def setup_2_512_128_split : Inst := mk 1024 128 true t1
theorem setup_2_512_128_split_ok : setup_2_512_128_split.ok :=
  mk_ok _ _ _ _ tok_t1 (by norm_num) (by norm_num)
def setup_3_2048_1200_split : Inst := mk 6144 1200 true t2
theorem setup_3_2048_1200_split_ok : setup_3_2048_1200_split.ok :=
  mk_ok _ _ _ _ tok_t2 (by norm_num) (by norm_num)
def setup_63 : Inst := mk 1 63 false t1
theorem setup_63_ok : setup_63.ok := mk_ok _ _ _ _ tok_t1 (by norm_num) (by norm_num)

def setup_gla_gk : Inst := mk 8192 1024 false t16
theorem setup_gla_gk_ok : setup_gla_gk.ok := mk_ok _ _ _ _ tok_t16 (by norm_num) (by norm_num)
def setup_gla_gate : Inst := mk 8192 2048 false t1
theorem setup_gla_gate_ok : setup_gla_gate.ok := mk_ok _ _ _ _ tok_t1 (by norm_num) (by norm_num)
def setup_long : Inst := mk 65536 4096 false t1
theorem setup_long_ok : setup_long.ok := mk_ok _ _ _ _ tok_t1 (by norm_num) (by norm_num)

end Claims.LogsigmoidFwd
