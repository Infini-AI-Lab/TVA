import TVA

/-!
# `matmul_f16`: the claim's definitions

Source: `source/03-matrix-multiplication.py` (tutorial 03, with the config of `SETUP.md`). The
kernel is `matmul_kernel` (lines 228–316); it reads the module-level names `triton` (its
decorator) and `tl` (lines 154–155). Its host function `matmul` (lines 330–350) launches it on
`float16` matrices: the test (lines 359–370, `512 × 512 × 512`) and the benchmark (lines 400–440,
square sizes `256, 384, …, 4096`).

This module holds the model (`matmul_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`) and the assumptions
(`Assumptions`). The proof is in `Matmul.Program` and `Matmul`.
-/

namespace Kernels.Matmul

open TVA

/-! ## The model -/

/-- `matmul_kernel` (source lines 228–316) with `ACTIVATION = ""`, the only value the host passes
(`matmul`'s default `activation=""`; the script never gives another): the compile-time condition
`ACTIVATION == "leaky_relu"` of line 306 is then false, the front end does not visit line 307, and
the model has no `ACTIVATION` parameter.

The pointers are `*fp16` (the host passes `float16` tensors). `M`, `N`, `K` and the six strides are
`int32`: an integer argument in `[-2^31, 2^31)` other than `1` is a runtime `int32`, and each is
below `2^31` on covered instances (`Inst.ok`); at `1` an argument is the compile-time constant `1`,
which gives the same model (see `model`). The constexprs `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`BLOCK_SIZE_K`, `GROUP_SIZE_M` are Lean parameters. -/
def matmul_kernel (a_ptr b_ptr c_ptr : Ptr .f16) (M N K : I32)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : Kernel where
  body ctx := do
    -- line 252: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 253: `tl.cdiv(M, BLOCK_SIZE_M)` is `(M + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: the
    -- constexpr `BLOCK_SIZE_M - 1` is computed by Python, then built in `int32` (an operand of `+`
    -- with the `int32` `M`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_m := (← tl.idiv (Tensor.scalar (M + (IntV.ofNat (BLOCK_SIZE_M - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_M : I32))).item
    -- line 254: the same with `N`, `BLOCK_SIZE_N`
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 255: the constexpr times the `int32` scalar, built in `int32`; wrapping `*`
    let num_pid_in_group := (IntV.ofNat GROUP_SIZE_M : I32) * num_pid_n
    -- line 256: `int32` `//` (`tl.idiv`)
    let group_id := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_in_group)).item
    -- line 257: `int32` scalar times the constexpr, built in `int32`
    let first_pid_m := group_id * (IntV.ofNat GROUP_SIZE_M : I32)
    -- line 258: Python's `min` with a tensor argument is `tl.minimum`, which makes the constexpr
    -- an `int32` tensor of its own type; on `int32` it is the exact `tl.iminimum`
    let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
      (Tensor.scalar (IntV.ofNat GROUP_SIZE_M : I32))).item
    -- line 259: two `int32` `%` (`tl.irem`), the inner one first, then the wrapping `+`
    let pid_m := first_pid_m + (← tl.irem (Tensor.scalar (← tl.irem (Tensor.scalar pid)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- line 260: `pid % num_pid_in_group` computed again, then `int32` `//`
    let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar pid)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- lines 266–273: `tl.assume(c)` is a check (undefined behaviour if false); each `c` is a
    -- comparison with the Python `0`, an `int32` tensor of its own type, signed; `x >= 0` is
    -- `0 <= x` and `x > 0` is `0 < x`
    tl.assume (IntV.le (IntV.ofInt 0 : I32) pid_m)
    tl.assume (IntV.le (IntV.ofInt 0 : I32) pid_n)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_am)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_ak)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_bn)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_bk)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_cm)
    tl.assume (IntV.lt (IntV.ofInt 0 : I32) stride_cn)
    -- line 282: the `int32` scalar `pid_m * BLOCK_SIZE_M` added to every lane of `tl.arange`, then
    -- `%` by the `int32` scalar `M`, broadcast (`tl.irem`)
    let offs_am : Tensor [BLOCK_SIZE_M] I32 ←
      tl.irem (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M) (bc M)
    -- line 283: the same with `pid_n`, `BLOCK_SIZE_N`, `N`
    let offs_bn : Tensor [BLOCK_SIZE_N] I32 ←
      tl.irem (pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N) (bc N)
    -- line 284
    let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K
    -- line 285: `offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak` in `int32`
    -- (wrapping), broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`, then added to the `*fp16` pointer
    let a_ptrs := a_ptr + (offs_am.col BLOCK_SIZE_K * stride_am + offs_k.row BLOCK_SIZE_M * stride_ak)
    -- line 286: the same for `b`, `[BLOCK_SIZE_K, BLOCK_SIZE_N]`
    let b_ptrs := b_ptr + (offs_k.col BLOCK_SIZE_N * stride_bk + offs_bn.row BLOCK_SIZE_K * stride_bn)
    -- line 293: `+0.0` in every element
    let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
    -- line 294: `range(0, tl.cdiv(K, BLOCK_SIZE_K))`: the bound `(K + (BLOCK_SIZE_K - 1)) //
    -- BLOCK_SIZE_K` is evaluated once, before the loop; `int32` bounds and induction variable; the
    -- loop carries `a_ptrs`, `b_ptrs`, `accumulator` (assigned in the body, defined before it)
    let (_a_ptrs, _b_ptrs, accumulator) ← tl.forRange (IntV.ofInt 0 : I32)
        (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
          (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
        (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, accumulator) fun k (a_ptrs, b_ptrs, accumulator) => do
      -- line 297: the mask `offs_k[None, :] < K - k * BLOCK_SIZE_K` (`int32`, signed), broadcast to
      -- the pointers' shape; `other=0.0` is the `float32` `+0.0` (a tensor of its own type),
      -- broadcast and converted to the pointer's dtype `float16` (`tl.convert`)
      let a ← tl.load a_ptrs (some ((offs_k.row BLOCK_SIZE_M).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))))
        (some (← tl.convert .f32 .f16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] (FloatV.litF32 (FloatV.ofPat 0)))))
      -- line 298: the mask `offs_k[:, None] < K - k * BLOCK_SIZE_K`, and `other=0.0` as above
      let b ← tl.load b_ptrs (some ((offs_k.col BLOCK_SIZE_N).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))))
        (some (← tl.convert .f32 .f16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] (FloatV.litF32 (FloatV.ofPat 0)))))
      -- line 300: `tl.dot(a, b, accumulator)` of `float16` operands into the `float32` accumulator;
      -- no `input_precision` (the default `tf32`) and no `max_num_imprecise_acc` (`0`)
      let accumulator ← tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
        a b accumulator
      -- lines 302–303: the `int32` product `BLOCK_SIZE_K * stride_ak` (the constexpr built in
      -- `int32`), added to every pointer of the block
      let a_ptrs := a_ptrs + (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_ak) : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I32)
      let b_ptrs := b_ptrs + (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_bk) : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I32)
      return (a_ptrs, b_ptrs, accumulator)
    -- lines 306–307: not visited (`ACTIVATION = ""`, see the docstring)
    -- line 308: `float32` to `float16`, one rounding to nearest even (`tl.cvt`)
    let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F16 ← tl.cvt accumulator
    -- lines 312–313
    let offs_cm : Tensor [BLOCK_SIZE_M] I32 := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
    let offs_cn : Tensor [BLOCK_SIZE_N] I32 := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- line 314: `(c_ptr + stride_cm * offs_cm[:, None]) + stride_cn * offs_cn[None, :]`: two
    -- pointer additions, each of an `int32` product, broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_N]`
    let c_ptrs := c_ptr + stride_cm * offs_cm.col BLOCK_SIZE_N + stride_cn * offs_cn.row BLOCK_SIZE_M
    -- line 315: two signed `int32` comparisons with the scalars `M`, `N`, broadcast, then `&`
    let c_mask := ((offs_cm.col BLOCK_SIZE_N).ltS M).and ((offs_cn.row BLOCK_SIZE_M).ltS N)
    -- line 316: `c` is already `float16`, the pointer's dtype
    tl.store c_ptrs c (some c_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `matmul_kernel` through `matmul(a, b)`. -/
structure Inst where
  /-- `M`: the rows of `a` and of `c` (`a.shape[0]`). -/
  M : Nat
  /-- `N`: the columns of `b` and of `c` (`b.shape[1]`). -/
  N : Nat
  /-- `K`: the columns of `a`, the rows of `b` (`a.shape[1]`). -/
  K : Nat
  /-- `stride_am` (`a.stride(0)`), in elements. -/
  stride_am : Nat
  /-- `stride_ak` (`a.stride(1)`). -/
  stride_ak : Nat
  /-- `stride_bk` (`b.stride(0)`). -/
  stride_bk : Nat
  /-- `stride_bn` (`b.stride(1)`). -/
  stride_bn : Nat
  /-- `stride_cm` (`c.stride(0)`). -/
  stride_cm : Nat
  /-- `stride_cn` (`c.stride(1)`). -/
  stride_cn : Nat
  /-- `a_ptr`: the storage of `a`, `float16`; element `(i, k)` of `a` is element
  `i * stride_am + k * stride_ak` of it. -/
  a : Buf .f16
  /-- `b_ptr`: the storage of `b`, `float16`; element `(k, j)` is `k * stride_bk + j * stride_bn`. -/
  b : Buf .f16
  /-- `c_ptr`: the storage of `c`, `float16`; element `(i, j)` is `i * stride_cm + j * stride_cn`. -/
  c : Buf .f16

/-- The elements of `a`'s storage that make up the matrix `a` (`M × K`). -/
def aIdx (d : Inst) : Set Nat :=
  {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.stride_am + k * d.stride_ak}

/-- The elements of `b`'s storage that make up the matrix `b` (`K × N`). -/
def bIdx (d : Inst) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.N ∧ e = k * d.stride_bk + j * d.stride_bn}

/-- The elements of `c`'s storage that make up the matrix `c` (`M × N`). -/
def cIdx (d : Inst) : Set Nat :=
  {e | ∃ i j, i < d.M ∧ j < d.N ∧ e = i * d.stride_cm + j * d.stride_cn}

/-- The instances covered. Shapes and strides are variables; the bounds keep the kernel's `int32`
arithmetic from wrapping, the tolerance meaningful (`K ≤ 2^13`), and the rows of `c` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and `tl.cdiv(M, 128)` computes `M + 127` in `int32`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 256 ≤ 2^31`: likewise for `tl.cdiv(N, 256)`. -/
  N_le : d.N + 256 ≤ 2 ^ 31
  /-- `K ≤ 2^13`: the tolerance of `Post` grows with `K`; up to `2^13` it is below `0.21%` of
  `Σ|a·b|` (`tol_rel_lt`), about half the largest relative error of a rounding to `bfloat16`
  (`2^-8/(1 + 2^-8)`, about `0.39%`), so `Post` still rejects a result rounded to `bfloat16`. From
  about `K = 15745` on it would not. The host's largest `K` is `4096`. -/
  K_le : d.K ≤ 2 ^ 13
  /-- Each stride is positive (`tl.assume(stride_… > 0)`, lines 268–273) and an `int32`. -/
  am : 0 < d.stride_am ∧ d.stride_am < 2 ^ 31
  /-- As `am`. -/
  ak : 0 < d.stride_ak ∧ d.stride_ak < 2 ^ 31
  /-- As `am`. -/
  bk : 0 < d.stride_bk ∧ d.stride_bk < 2 ^ 31
  /-- As `am`. -/
  bn : 0 < d.stride_bn ∧ d.stride_bn < 2 ^ 31
  /-- As `am`. -/
  cm : 0 < d.stride_cm ∧ d.stride_cm < 2 ^ 31
  /-- As `am`. -/
  cn : 0 < d.stride_cn ∧ d.stride_cn < 2 ^ 31
  /-- The `int32` offsets of `a_ptrs` (line 285), `i * stride_am + k * stride_ak` for `i < M`,
  `k < 64`, and the step `64 * stride_ak` (line 302), do not wrap. -/
  a_fit : d.M * d.stride_am + 64 * d.stride_ak ≤ 2 ^ 31
  /-- The `int32` offsets of `b_ptrs` (line 286) and the step `64 * stride_bk` do not wrap. -/
  b_fit : 64 * d.stride_bk + d.N * d.stride_bn ≤ 2 ^ 31
  /-- The `int32` products `stride_cm * offs_cm` of line 314 do not wrap on the rows below `M`. -/
  cm_fit : d.M * d.stride_cm < 2 ^ 31
  /-- Each row of `c` fits in `stride_cm`: a row's `N` elements, `stride_cn` apart, lie within
  `stride_cm` (row-major, possibly padded; `c` as `matmul` allocates it has `stride_cm = N`,
  `stride_cn = 1`), so the rows do not overlap. Column-major and other transposed layouts of `c`
  are excluded too, though their rows do not overlap. -/
  c_rows : d.N * d.stride_cn ≤ d.stride_cm
  /-- `a`'s storage holds every element of the matrix `a`. -/
  a_len : ∀ i k, i < d.M → k < d.K → i * d.stride_am + k * d.stride_ak < d.a.len
  /-- `b`'s storage holds every element of the matrix `b`. -/
  b_len : ∀ k j, k < d.K → j < d.N → k * d.stride_bk + j * d.stride_bn < d.b.len
  /-- `c`'s storage holds every element of the matrix `c`. -/
  c_len : ∀ i j, i < d.M → j < d.N → i * d.stride_cm + j * d.stride_cn < d.c.len
  /-- The matrix `c` shares no byte with the matrix `a`. -/
  c_a : Disjoint (d.c.elems (cIdx d)) (d.a.elems (aIdx d))
  /-- The matrix `c` shares no byte with the matrix `b`. -/
  c_b : Disjoint (d.c.elems (cIdx d)) (d.b.elems (bIdx d))

/-- The model on an instance: the buffers' first elements, `M`, `N`, `K` and the strides as
`int32`, and the block sizes of the launch (line 346): `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`,
`BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`. An integer argument equal to `1` is the compile-time
constant `1`; every use of `M`, `N`, `K` and of a stride then makes it an `int32` (an operand of
`+`, `-`, `*`, `%` or a comparison with an `int32` tensor builds it in `int32`; `tl.cdiv(1, B)` of
two constants is Python's `1`, which the name it is assigned to holds as an `int32` tensor, as the
runtime `int32` `1` gives; at `K = 1`, `tl.cdiv(K, BLOCK_SIZE_K)` of line 294 is Python's `1`, which
goes straight to `range(0, 1)`: the loop's bounds `0`, `1` and its step are then three Python
integers, so the induction variable is an `int32` and the loop runs one trip, as the model's
`tl.forRange` from `0` to the `int32` `(1 + 63) // 64 = 1` gives; `stride > 0` is Python's `True`, which `tl.assume` makes the `int1`
`true`, as the `int32` comparison gives; `BLOCK_SIZE_K * stride` is Python's `64`, which the pointer
addition makes an `int32`), so the model is the same. -/
def model (d : Inst) : Kernel :=
  matmul_kernel (d.a.ptr 0) (d.b.ptr 0) (d.c.ptr 0) (IntV.ofNat d.M) (IntV.ofNat d.N) (IntV.ofNat d.K)
    (IntV.ofNat d.stride_am) (IntV.ofNat d.stride_ak) (IntV.ofNat d.stride_bk) (IntV.ofNat d.stride_bn)
    (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn) 128 256 64 8

/-- The host's launch (lines 339–349): a 1-D grid of `triton.cdiv(M, 128) * triton.cdiv(N, 256)`
programs; the source's block shapes; `enable_fp_fusion` is Triton's default (`true`); `grid1`
keeps `num_ctas = 1`, `launch_pdl = False` and `debug = false`. `num_warps=8` and `num_stages=3`
are not recorded (performance-only, and harmless to a well-behaved model). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256))
    [[128], [256], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 256], [64, 256], [128, 256]]
    (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present). Any `float16` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.c.Valid init

/-! ## The specification -/

/-- The number of `tl.dot` steps of the loop (line 294): `cdiv(K, 64)`. -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- The tolerance of the `float32` accumulation, for a reduction length `K` and the sum `A` of the
products' lifted magnitudes (`dotMag`): `DotLoop.bound ρ α T A = ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, with
`T = steps K` `tl.dot` steps of `64` products each, and `ρ = relDot16 64 = 66·2^-22`,
`α = absDot16 64 = 66·2^-126`, the library's constants for one fp16 `tl.dot` of `K = 64`
(`Rules.Num.DotF16`). It is about `1.0e-3·A` at `K = 4096` (the host's largest) and `2.0e-3·A` at
`K = 2^13` (the largest covered, `tol_rel_lt`), plus at most `10^-34`; and it is at most
`2T·(ρ·A + α)`. -/
def tol (K : Nat) (A : ℚ) : ℚ :=
  DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) (steps K) A

/-- The relative part of `tol` on covered instances (`K ≤ 2^13`, `Inst.ok.K_le`): below `0.21%`,
about half the largest relative error of a rounding to `bfloat16` (`2^-8/(1 + 2^-8)`). -/
theorem tol_rel_lt {K : Nat} (hK : K ≤ 2 ^ 13) :
    (1 + Rules.Num.relDot16 64) ^ steps K - 1 < 21 / 10000 := by
  have hT : steps K ≤ 128 := by unfold steps; omega
  have h1 : (1 : ℚ) ≤ 1 + Rules.Num.relDot16 64 := by unfold Rules.Num.relDot16; norm_num
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : (1 + Rules.Num.relDot16 64) ^ 128 < 1 + 21 / 10000 := by
    unfold Rules.Num.relDot16; norm_num
  linarith

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The magnitude by which the tensor core aligns a `float16` operand value `q`: `0` for `0`, and
otherwise `|q|`, or the smallest normal magnitude `2^-14` when `|q|` is below it (a subnormal). It is
`|q|` for zero and normal values. The library's `Rules.Num.liftMag .f16`, which `Rules.Num.DotF16`
counts its magnitudes by (`liftMag_f16`; library issue L126). -/
def mag16 (q : ℚ) : ℚ := if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ))

/-- The sum of the products' lifted magnitudes, `Σ mag16 (va k) · mag16 (vb k)` over `k < K`. With
no subnormal operand it is `Σ |va k · vb k|`; it is at most `Σ |va k · vb k| + 2^-14·(Σ |va k| +
Σ |vb k|) + K·2^-28` (`DotLift.liftMag_mul_le`). -/
def dotMag (va vb : Nat → F16) (K : Nat) : ℚ :=
  ∑ k ∈ Finset.range K, mag16 (va k).toRat * mag16 (vb k).toRat

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is
the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` within
`tol K A` of the exact sum `S = Σ va k · vb k` (`A = dotMag`, `Σ |va k · vb k|` when no operand is
subnormal, a subnormal operand counted at `2^-14`); and when there is room in
`float16` (`|S| + tol K A ≤ 65504`), `c` is finite and within `tol K A` plus one `float16`
rounding (`(|S| + tol K A)·2^-11 + 2^-25`) of `S`. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum va vb d.K| ≤ tol d.K (dotMag va vb d.K) ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum va vb d.K| + tol d.K (dotMag va vb d.K) ≤ Math.Float.Format.binary16.maxVal →
          c.isFinite = true ∧
          |c.toRat - dotSum va vb d.K| ≤ tol d.K (dotMag va vb d.K) +
            (|dotSum va vb d.K| + tol d.K (dotMag va vb d.K)) * Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta)

/-- What the launch leaves: every element of the matrix `c` as `OutSpec` says; every byte outside
the matrix `c` unchanged (`a`, `b`, and the elements of `c`'s storage outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i j, i < d.M → j < d.N → OutSpec d init fin i j) ∧
    ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotF16`: `tl.dot` of `float16` operands into `float32` (`maxImpreciseAcc = 0`,
  any `input_precision`, the model's line 300; `Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at the smallest normal `2^-14`), when that leaves room below the
  `float32` range. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 300) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

end Kernels.Matmul
