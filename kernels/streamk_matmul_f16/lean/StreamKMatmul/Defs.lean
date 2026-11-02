import TVA

/-!
# `streamk_matmul_f16`: the claim's definitions

Source: `source/streamk_matmul.py`, written for this folder (`SETUP.md`). The kernel is
`streamk_matmul_kernel` (lines 26–64); it reads the module-level names `triton` (its decorator) and
`tl` (lines 22–23). Its host function `streamk_matmul` (lines 70–93) launches it on `float16`
matrices `a` (`M × K`) and `b` (`K × N`) into a new `float32` matrix `c = torch.zeros(M, N)`, with a
1-D grid of `num_programs` programs (the SM count unless the caller passes one).

Stream-K: the `num_tiles · iters_per_tile` iterations of the `K` loop of all tiles are split into
contiguous ranges of `iters_per_program`, one per program. A tile one program owns entirely is
stored; a tile split across programs is accumulated with `tl.atomic_add` by each program that owns
part of it.

This module holds the model (`streamk_matmul_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`tol`, `OutSpec`, `Post`, `Spec`) and the assumptions
(`Assumptions`). The proof is in the other modules of `StreamKMatmul`.
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## The model -/

/-- `streamk_matmul_kernel` (source lines 26–64).

The pointers `a_ptr`, `b_ptr` are `*fp16` and `c_ptr` is `*fp32` (the host passes `float16` `a`,
`b` and a `float32` `c`). `M`, `N`, `K`, the six strides, `iters_per_program` and `total_iters` are
`int32`: an integer argument in `[-2^31, 2^31)` other than `1` is a runtime `int32`, and each is
below `2^31` on covered instances (`Inst.ok`); at `1` an argument is the compile-time constant `1`,
which gives the same model (see `model`). The constexprs `BLOCK_M`, `BLOCK_N`, `BLOCK_K` are Lean
parameters.

The source's name `end` (line 40) is a keyword of Lean, so the model writes it `«end»`, the same
name. -/
def streamk_matmul_kernel (a_ptr b_ptr : Ptr .f16) (c_ptr : Ptr .f32) (M N K : I32)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : I32)
    (iters_per_program total_iters : I32) (BLOCK_M BLOCK_N BLOCK_K : Nat) : Kernel where
  body ctx := do
    -- line 36: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 37: `tl.cdiv(N, BLOCK_N)` is `(N + (BLOCK_N - 1)) // BLOCK_N`: the constexpr
    -- `BLOCK_N - 1` is computed by Python, then built in `int32` (an operand of `+` with the
    -- `int32` `N`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_N : I32))).item
    -- line 38: the same with `K`, `BLOCK_K`
    let iters_per_tile := (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_K - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_K : I32))).item
    -- line 39: the wrapping `int32` product of two `int32` scalars
    let it := pid * iters_per_program
    -- line 40: `tl.minimum` of two `int32` scalars is the exact signed `tl.iminimum`; the
    -- wrapping `int32` `+`
    let «end» := (tl.iminimum (Tensor.scalar (it + iters_per_program)) (Tensor.scalar total_iters)).item
    -- line 41: `while it < end:` with the runtime condition `it < end` (a signed `int32`
    -- comparison); the loop carries `it`, the one name defined before it that its body assigns
    -- (line 64); the names lines 42–63 define are new to the body and not used after the loop
    let _it ← tl.whileLoop it (fun it => pure (IntV.lt it «end»)) fun it => do
      -- line 42: `int32` `//` (`tl.idiv`)
      let tile := (← tl.idiv (Tensor.scalar it) (Tensor.scalar iters_per_tile)).item
      -- line 44: wrapping `int32` `*` and `-`
      let k_first := it - tile * iters_per_tile
      -- line 45: `tl.minimum` on `int32` (`tl.iminimum`); the wrapping `-`, then `+`
      let k_last := (tl.iminimum (Tensor.scalar iters_per_tile) (Tensor.scalar (k_first + («end» - it)))).item
      -- line 46: `int32` `//`
      let pid_m := (← tl.idiv (Tensor.scalar tile) (Tensor.scalar num_pid_n)).item
      -- line 47: `int32` `%` (`tl.irem`)
      let pid_n := (← tl.irem (Tensor.scalar tile) (Tensor.scalar num_pid_n)).item
      -- line 48: the `int32` scalar `pid_m * BLOCK_M` (the constexpr built in `int32`) added to
      -- every lane of `tl.arange`
      let offs_m : Tensor [BLOCK_M] I32 := pid_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
      -- line 49: the same with `pid_n`, `BLOCK_N`
      let offs_n : Tensor [BLOCK_N] I32 := pid_n * (IntV.ofNat BLOCK_N : I32) + tl.arange 0 BLOCK_N
      -- line 50: `+0.0` in every element
      let acc := tl.zeros [BLOCK_M, BLOCK_N] .f32
      -- line 51: `range(k_first, k_last)`: two `int32` bounds, step `1`, an `int32` induction
      -- variable; the loop carries `acc` (defined before it, line 50, and assigned in the body)
      let acc ← tl.forRange k_first k_last (IntV.ofInt 1 : I32) acc fun kk acc => do
        -- line 52: the `int32` scalar `kk * BLOCK_K` added to every lane of `tl.arange`
        let offs_k : Tensor [BLOCK_K] I32 := kk * (IntV.ofNat BLOCK_K : I32) + tl.arange 0 BLOCK_K
        -- lines 53–54: `(a_ptr + offs_m[:, None] * stride_am) + offs_k[None, :] * stride_ak`: two
        -- pointer additions, each of an `int32` product, broadcast to `[BLOCK_M, BLOCK_K]`; the
        -- mask: two signed `int32` comparisons with the scalars `M`, `K`, broadcast, then `&`;
        -- `other=0.0` is the `float32` `+0.0` (a tensor of its own type), broadcast and converted
        -- to the pointer's dtype `float16` (`tl.convert`)
        let a ← tl.load (a_ptr + offs_m.col BLOCK_K * stride_am + offs_k.row BLOCK_M * stride_ak)
          (some (((offs_m.col BLOCK_K).ltS M).and ((offs_k.row BLOCK_M).ltS K)))
          (some (← tl.convert .f32 .f16 (tl.full [BLOCK_M, BLOCK_K] (FloatV.litF32 (FloatV.ofPat 0)))))
        -- lines 55–56: the same for `b`, `[BLOCK_K, BLOCK_N]`
        let b ← tl.load (b_ptr + offs_k.col BLOCK_N * stride_bk + offs_n.row BLOCK_K * stride_bn)
          (some (((offs_k.col BLOCK_N).ltS K).and ((offs_n.row BLOCK_K).ltS N)))
          (some (← tl.convert .f32 .f16 (tl.full [BLOCK_K, BLOCK_N] (FloatV.litF32 (FloatV.ofPat 0)))))
        -- line 57: `tl.dot(a, b, acc)` of `float16` operands into the `float32` accumulator; no
        -- `input_precision` (the default `tf32`) and no `max_num_imprecise_acc` (`0`)
        let acc ← tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
          a b acc
        return acc
      -- line 58: `(c_ptr + offs_m[:, None] * stride_cm) + offs_n[None, :] * stride_cn`, as lines
      -- 53–54, broadcast to `[BLOCK_M, BLOCK_N]`
      let c_ptrs := c_ptr + offs_m.col BLOCK_N * stride_cm + offs_n.row BLOCK_M * stride_cn
      -- line 59: two signed `int32` comparisons with the scalars `M`, `N`, broadcast, then `&`
      let c_mask := ((offs_m.col BLOCK_N).ltS M).and ((offs_n.row BLOCK_M).ltS N)
      -- line 60: a runtime `if`; each `==` compares `int32` scalars (the Python `0` built as an
      -- `int32` tensor of its own type) and gives an `int1`, and `&` of two `int1` is their `and`
      if IntV.eq k_first (IntV.ofInt 0 : I32) && IntV.eq k_last iters_per_tile then
        -- line 61: `acc` is already `float32`, the pointer's dtype
        tl.store c_ptrs acc (some c_mask)
      else
        -- line 63: the defaults `sem="acq_rel"`, `scope="gpu"`; the result is not used
        let _ ← tl.atomic_add c_ptrs acc (some c_mask)
        pure ()
      -- line 64: `it += k_last - k_first`, two wrapping `int32` operations
      return it + (k_last - k_first)
    -- the loop is the last statement: its result is not used
    pure ()

/-! ## The conditions -/

/-- What the host chooses: one launch of `streamk_matmul_kernel` through
`streamk_matmul(a, b, num_programs)`. -/
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
  /-- `num_programs`: the size of the 1-D grid (line 82). -/
  num_programs : Nat
  /-- The argument `iters_per_program` (line 81: `cdiv(total_iters, num_programs)`). -/
  iters_per_program : Nat
  /-- The argument `total_iters` (line 80: `num_tiles * cdiv(K, BLOCK_K)`). -/
  total_iters : Nat
  /-- `a_ptr`: the storage of `a`, `float16`; element `(i, k)` of `a` is element
  `i * stride_am + k * stride_ak` of it. -/
  a : Buf .f16
  /-- `b_ptr`: the storage of `b`, `float16`; element `(k, j)` is `k * stride_bk + j * stride_bn`. -/
  b : Buf .f16
  /-- `c_ptr`: the storage of `c`, `float32`; element `(i, j)` is `i * stride_cm + j * stride_cn`. -/
  c : Buf .f32

/-- The number of `tl.dot` steps of a whole tile: `iters_per_tile = cdiv(K, 64)` (line 38). -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- The tile rows, `cdiv(M, 128)`. -/
def tilesM (d : Inst) : Nat := (d.M + 127) / 128

/-- The tile columns, `num_pid_n = cdiv(N, 128)` (line 37). -/
def tilesN (d : Inst) : Nat := (d.N + 127) / 128

/-- The elements of `a`'s storage that make up the matrix `a` (`M × K`). -/
def aIdx (d : Inst) : Set Nat :=
  {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.stride_am + k * d.stride_ak}

/-- The elements of `b`'s storage that make up the matrix `b` (`K × N`). -/
def bIdx (d : Inst) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.N ∧ e = k * d.stride_bk + j * d.stride_bn}

/-- The elements of `c`'s storage that make up the matrix `c` (`M × N`). -/
def cIdx (d : Inst) : Set Nat :=
  {e | ∃ i j, i < d.M ∧ j < d.N ∧ e = i * d.stride_cm + j * d.stride_cn}

/-- The instances covered. Shapes, strides and program counts are variables; the bounds keep the
kernel's `int32` arithmetic from wrapping, the tolerance meaningful (`K ≤ 2^13`), and the rows of
`c` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M`, `N`, `K` are positive: `streamk_matmul` asserts it (line 75). -/
  M_pos : 0 < d.M
  /-- As `M_pos`. -/
  N_pos : 0 < d.N
  /-- As `M_pos`. -/
  K_pos : 0 < d.K
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and the tile rows `pid_m * 128 + i` (line 48) stay
  below `cdiv(M, 128) * 128 ≤ M + 127`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 128 ≤ 2^31`: `N` is an `int32`, `tl.cdiv(N, 128)` computes `N + 127` in `int32`
  (line 37), and the tile columns stay below `N + 127` (line 49). -/
  N_le : d.N + 128 ≤ 2 ^ 31
  /-- `K ≤ 2^13`: the tolerance of `Post` grows with `K`; up to `2^13` it is below `0.21%` of
  `A` (`dotMag`; `Σ|a·b|` when no operand is subnormal) (`tol_le`), about half the largest
  relative error of a rounding to `bfloat16` (`2^-8/(1 + 2^-8)`, about `0.39%`), so `Post` still
  rejects a result rounded to `bfloat16`. The test's largest `K` is `4097`. It also makes `K + 63 < 2^31` for `tl.cdiv(K, 64)` (line 38). -/
  K_le : d.K ≤ 2 ^ 13
  /-- Each stride is an `int32` (below `2^31`). -/
  am : d.stride_am < 2 ^ 31
  /-- As `am`. -/
  ak : d.stride_ak < 2 ^ 31
  /-- As `am`. -/
  bk : d.stride_bk < 2 ^ 31
  /-- As `am`. -/
  bn : d.stride_bn < 2 ^ 31
  /-- As `am`. -/
  cm : d.stride_cm < 2 ^ 31
  /-- As `am`, and positive: with `c_rows`, distinct elements of the matrix `c` are distinct
  elements of its storage. -/
  cn : 0 < d.stride_cn ∧ d.stride_cn < 2 ^ 31
  /-- The `int32` products `offs_m * stride_am` of the rows below `M` (line 53) do not wrap. -/
  am_fit : (d.M - 1) * d.stride_am < 2 ^ 31
  /-- The `int32` products `offs_k * stride_ak` of the columns below `K` (line 53) do not wrap. -/
  ak_fit : (d.K - 1) * d.stride_ak < 2 ^ 31
  /-- The `int32` products `offs_k * stride_bk` of the rows below `K` (line 55) do not wrap. -/
  bk_fit : (d.K - 1) * d.stride_bk < 2 ^ 31
  /-- The `int32` products `offs_n * stride_bn` of the columns below `N` (line 55) do not wrap. -/
  bn_fit : (d.N - 1) * d.stride_bn < 2 ^ 31
  /-- The `int32` products `offs_m * stride_cm` of the rows below `M` (line 58) do not wrap. -/
  cm_fit : (d.M - 1) * d.stride_cm < 2 ^ 31
  /-- Each row of `c` fits in `stride_cm`: a row's `N` elements, `stride_cn` apart, lie within
  `stride_cm` (row-major, possibly padded; `c` as `streamk_matmul` allocates it has
  `stride_cm = N`, `stride_cn = 1`), so the rows do not overlap and the products
  `offs_n * stride_cn` of the columns below `N` (line 58) do not wrap. Column-major and other
  transposed layouts of `c` are excluded too, though their rows do not overlap. -/
  c_rows : d.N * d.stride_cn ≤ d.stride_cm
  /-- `total_iters` is the host's `num_tiles * cdiv(K, BLOCK_K)` (lines 79–80). -/
  total : d.total_iters = tilesM d * tilesN d * steps d.K
  /-- The programs' ranges cover every iteration: `total_iters ≤ num_programs * iters_per_program`,
  as `iters_per_program = cdiv(total_iters, num_programs)` (line 81) gives. -/
  cover : d.total_iters ≤ d.num_programs * d.iters_per_program
  /-- `num_programs * iters_per_program < 2^31`: the `int32` values `pid * iters_per_program` and
  `it + iters_per_program` (lines 39–40) do not wrap, and the grid has fewer than `2^31`
  programs. -/
  progs_lt : d.num_programs * d.iters_per_program < 2 ^ 31
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

/-- The model on an instance: the buffers' first elements, `M`, `N`, `K`, the strides,
`iters_per_program` and `total_iters` as `int32`, and the block sizes of the launch (line 90):
`BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`.

An integer argument equal to `1` is the compile-time constant `1`; every use of such an argument
then gives the model's `int32` value: an operand of `+`, `*` or a comparison with an `int32` tensor
is built in `int32` (lines 39, 40, 53–56, 58, 59), an argument of `tl.minimum` becomes an `int32`
tensor of its own type (line 40), and `tl.cdiv(1, B)` of two constants is Python's `1`, which the
name it is assigned to (lines 37–38) holds as an `int32` tensor, as the runtime `int32` `1` gives
(`(1 + 127) // 128 = (1 + 63) // 64 = 1`). So the model is the same. -/
def model (d : Inst) : Kernel :=
  streamk_matmul_kernel (d.a.ptr 0) (d.b.ptr 0) (d.c.ptr 0) (IntV.ofNat d.M) (IntV.ofNat d.N) (IntV.ofNat d.K)
    (IntV.ofNat d.stride_am) (IntV.ofNat d.stride_ak) (IntV.ofNat d.stride_bk) (IntV.ofNat d.stride_bn)
    (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn) (IntV.ofNat d.iters_per_program)
    (IntV.ofNat d.total_iters) 128 128 64

/-- The host's launch (lines 82–92): a 1-D grid of `num_programs` programs; the source's block
shapes; `enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`,
`launch_pdl = False` and `debug = false`. `num_warps=8` and `num_stages=3` are not recorded
(performance-only, and harmless to a well-behaved model). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.num_programs
    [[128], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 128], [64, 128], [128, 128]]
    (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and every element of the matrix `c` holds a zero (`streamk_matmul` allocates `c` with
`torch.zeros`, line 78). Any `float16` values in `a` and `b`, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.c.Valid init ∧
    ∀ i j, i < d.M → j < d.N → ∃ z : F32,
      init.read .f32 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some z ∧
        z.isFinite = true ∧ z.toRat = 0

/-! ## The specification -/

/-- The accumulation's part of the tolerance: `DotLoop.bound ρ α T A = ((1 + ρ)^T − 1)·A +
T·(1 + ρ)^T·α`, with `T = steps K` `tl.dot` steps of `64` products each, and `ρ = relDot16 64 =
66·2^-22`, `α = absDot16 64 = 66·2^-126`, the library's constants for one fp16 `tl.dot` of `K = 64`
(`Rules.Num.DotF16`). -/
def tolDot (K : Nat) (A : ℚ) : ℚ :=
  DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) (steps K) A

/-- The tolerance, for a reduction length `K` and the sum `A` of the products' lifted magnitudes
(`dotMag`): `tolDot K A`, the accumulation of the `tl.dot` steps, plus
`DotLoop.bound 2^-23 2^-124 T (A + tolDot K A)`, at most `T` `float32` atomic additions of partial
sums whose magnitudes add up to at most `A + tolDot K A`, with `2^-23` and `2^-124` the constants of
`Rules.Num.AtomicAddF32`. It is about `1.03·10^-3·A` at `K = 4097` (the test's largest) and below
`2.1·10^-3·A` at `K = 2^13` (the largest covered, `tol_le`), plus at most `10^-33`. -/
def tol (K : Nat) (A : ℚ) : ℚ :=
  tolDot K A + DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (steps K) (A + tolDot K A)

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The magnitude by which the tensor core aligns a `float16` operand value `q`: `0` for `0`, and
otherwise `|q|`, or the smallest normal magnitude `2^-14` when `|q|` is below it (a subnormal). It is
`|q|` for zero and normal values. The library's `Rules.Num.liftMag .f16`, which `Rules.Num.DotF16`
counts its magnitudes by (`liftMag_f16`). -/
def mag16 (q : ℚ) : ℚ := if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ))

/-- The sum of the products' lifted magnitudes, `Σ mag16 (va k) · mag16 (vb k)` over `k < K`. With
no subnormal operand it is `Σ |va k · vb k|`. -/
def dotMag (va vb : Nat → F16) (K : Nat) : ℚ :=
  ∑ k ∈ Finset.range K, mag16 (va k).toRat * mag16 (vb k).toRat

/-- What the launch leaves in element `(i, j)` of `c`: some `float32` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is
finite and within `tol K A` of the exact sum `S = Σ va k · vb k` (`A = dotMag`, `Σ |va k · vb k|`
when no operand is subnormal), whatever order the atomic additions ran in. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F32, fin.read .f32 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      c.isFinite = true ∧ |c.toRat - dotSum va vb d.K| ≤ tol d.K (dotMag va vb d.K)

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
  any `input_precision`, the model's line 57; `Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at the smallest normal `2^-14`), when that leaves room below the
  `float32` range. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 57) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
  /-- [rule] `Num.AtomicAddF32`: `tl.atomic_add` on `float32` (line 63; `Rules.Num.AtomicAddAcc`
  with `2^-23`, `2^-124`): when the old value and the operand are finite and their exact sum is in
  range, the new value is finite and within `2^-23·|o + v| + 2^-124` of the sum. -/
  atomic : Rules.Num.AtomicAddF32 P
  /-- [rule] `Num.AtomicAddTotal`: a `float32` `tl.atomic_add` (line 63) always has a result. -/
  atomic_total : Rules.Num.AtomicAddTotal P Math.Float.Format.binary32

end Kernels.StreamKMatmul
