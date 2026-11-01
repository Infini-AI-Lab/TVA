import TVA

/-!
# `matmul_e5m2`: the claim's definitions

Source: `source/03-matrix-multiplication.py` (tutorial 03, with the config of `SETUP.md`). The
kernel is `matmul_kernel` (lines 228–316); it reads the module-level names `triton` (its
decorator) and `tl` (lines 154–155). Its host function `matmul` (lines 330–350) launches it, in
the script's `float8e5` run, on `float8e5` matrices `a` (contiguous) and `b` (the transpose of a
contiguous matrix, strides `(1, K)`) into a new `float16` matrix `c`: the test (lines 372–388,
`512 × 512 × 512`) and the benchmark's `fp8` line (lines 400–440, square sizes
`256, 384, …, 4096`).

This module holds the model (`matmul_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`) and the assumptions
(`Assumptions`). The proof is in `MatmulE5m2.Program` and `MatmulE5m2`.
-/

namespace Kernels.MatmulE5m2

open TVA

/-! ## The model -/

/-- `matmul_kernel` (source lines 228–316) with `ACTIVATION = ""`, the only value the host passes
(`matmul`'s default `activation=""`; the script never gives another): the compile-time condition
`ACTIVATION == "leaky_relu"` of line 306 is then false, the front end does not visit line 307, and
the model has no `ACTIVATION` parameter.

The pointers `a_ptr` and `b_ptr` are `*fp8e5` and `c_ptr` is `*fp16` (the host passes
`float8e5` tensors `a`, `b` and the `float16` tensor `c` it allocates). `M`, `N`, `K` and the six
strides are `int32`: an integer argument in `[-2^31, 2^31)` other than `1` is a runtime `int32`,
and each is below `2^31` on covered instances (`Inst.ok`); at `1` an argument is the compile-time
constant `1`, which gives the same model (see `model`). The constexprs `BLOCK_SIZE_M`,
`BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M` are Lean parameters. -/
def matmul_kernel (a_ptr b_ptr : Ptr .f8e5) (c_ptr : Ptr .f16) (M N K : I32)
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
    -- (wrapping), broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`, then added to the `*fp8e5` pointer
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
      -- broadcast and converted to the pointer's dtype `float8e5` (`tl.convert`, on the device)
      let a ← tl.load a_ptrs (some ((offs_k.row BLOCK_SIZE_M).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))))
        (some (← tl.convert .f32 .f8e5 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] (FloatV.litF32 (FloatV.ofPat 0)))))
      -- line 298: the mask `offs_k[:, None] < K - k * BLOCK_SIZE_K`, and `other=0.0` as above
      let b ← tl.load b_ptrs (some ((offs_k.col BLOCK_SIZE_N).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))))
        (some (← tl.convert .f32 .f8e5 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] (FloatV.litF32 (FloatV.ofPat 0)))))
      -- line 300: `tl.dot(a, b, accumulator)` of `float8e5` operands into the `float32`
      -- accumulator; no `input_precision` (the default `tf32`) and no `max_num_imprecise_acc`
      -- (`2^30`, the default for an fp8 × fp8 dot on sm_90)
      let accumulator ← tl.dot { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 }
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

/-- What the host chooses: one launch of `matmul_kernel` through `matmul(a, b)` in the script's
`float8e5` run. -/
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
  /-- `a_ptr`: the storage of `a`, `float8e5`; element `(i, k)` of `a` is element
  `i * stride_am + k * stride_ak` of it. -/
  a : Buf .f8e5
  /-- `b_ptr`: the storage of `b`, `float8e5`; element `(k, j)` is `k * stride_bk + j * stride_bn`. -/
  b : Buf .f8e5
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
arithmetic from wrapping, `K` within the host's use (`K ≤ 4096`), and the rows of `c` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and `tl.cdiv(M, 128)` computes `M + 127` in `int32`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 256 ≤ 2^31`: likewise for `tl.cdiv(N, 256)`. -/
  N_le : d.N + 256 ≤ 2 ^ 31
  /-- `K ≤ 2^12 = 4096`, the host's largest `K` (the benchmark's `4096`). At every covered `K`
  the tolerance of `Post` rejects a zero result and a skipped `K`-step on some input
  (`zero_rejected`, `drop_rejected`, in `MatmulE5m2.Reject`); the proof also uses the bound to keep
  the partial sums far inside the `float32` range. Beyond the host's use the tolerance would only
  grow. -/
  K_le : d.K ≤ 2 ^ 12
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
constant `1` (on every launch of the script's `float8e5` run, `stride_ak = 1`, `stride_bk = 1` and
`stride_cn = 1`: `a` is contiguous, `b` is the transpose of a contiguous matrix, `c` is allocated
row-major); every use of `M`, `N`, `K` and of a stride then makes it an `int32` (an operand of
`+`, `-`, `*`, `%` or a comparison with an `int32` tensor builds it in `int32`; `tl.cdiv(1, B)` of
two constants is Python's `1`, which the name it is assigned to holds as an `int32` tensor, as the
runtime `int32` `1` gives; at `K = 1`, `tl.cdiv(K, BLOCK_SIZE_K)` of line 294 is Python's `1`, which
goes straight to `range(0, 1)`: the loop's bounds `0`, `1` and its step are then three Python
integers, so the induction variable is an `int32` and the loop runs one trip, as the model's
`tl.forRange` from `0` to the `int32` `(1 + 63) // 64 = 1` gives; `stride > 0` is Python's `True`,
which `tl.assume` makes the `int1` `true`, as the `int32` comparison gives; `BLOCK_SIZE_K * stride`
is Python's `64`, which the pointer addition makes an `int32`), so the model is the same. -/
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
present). Any `float8e5` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.c.Valid init

/-! ## The specification -/

/-- The number of `tl.dot` steps of the loop (line 294): `cdiv(K, 64)`. Each step's `tl.dot` of
`K = 64` adds two chunks of `32` products (`Rules.Num.DotE5m2Chunk`), so the loop adds
`2·steps K` chunks: chunk `j` holds the products `32·j ≤ e < 32·j + 32`. -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- Product `e` of the reduction of row values `va` by column values `vb`: `va e · vb e` for
`e < K`, and `0` from `K` on (the lanes the masks disable). -/
def prodV (va vb : Nat → F8E5) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The lifted magnitude of product `e`: `liftMag (va e) · liftMag (vb e)` for `e < K`, and `0`
from `K` on. `Rules.Num.liftMag` is `|x|` for a normal `float8e5` value or a zero, and `2^-14` (the
smallest normal magnitude) for a subnormal one: the magnitude by which the tensor core aligns the
product (`Rules.Num.DotE5m2Chunk`). It is `|prodV va vb K e|` when neither operand is subnormal. -/
def liftV (va vb : Nat → F8E5) (K e : Nat) : ℚ :=
  if e < K then Rules.Num.liftMag .f8e5 (va e).toRat * Rules.Num.liftMag .f8e5 (vb e).toRat else 0

/-- The exact partial sum after `j` chunks of a sequence of products `p`: `p e` for `e < 32·j`. -/
def partSum (p : Nat → ℚ) (j : Nat) : ℚ := ∑ e ∈ Finset.range (32 * j), p e

/-- The largest value of `q` on chunk `j`: `q e` for `32·j ≤ e < 32·j + 32`. -/
def chunkMax (q : Nat → ℚ) (j : Nat) : ℚ :=
  Finset.univ.sup' Finset.univ_nonempty fun k : Fin 32 => q (32 * j + k)

/-- **The error bound after `j` chunks**, from the exact partial sums: `E 0 = 0` and
`E (j + 1) = (1 + ρ + σ)·E j + ρ·max(|S j|, m j) + σ·|S (j + 1)| + α`, with `S j = partSum p j` the
exact partial sum before chunk `j`, `m j = chunkMax q j` the largest lifted product of chunk `j`,
and the library's constants for one chunk (`Rules.Num.DotE5m2Chunk`): `ρ = rhoChunk8 = 2^-7`,
`σ = sigmaChunk8 = 2^-12`, `α = alphaChunk8 = 2^-126`. It is what the fact gives chunk by chunk:
chunk `j` adds its exact sum `c` to an accumulator `s` within `E j` of `S j`, up to
`ρ·M + σ·|s + c| + α` for every `M` at least `|s|` and the chunk's lifted products; here
`M = max(|S j| + E j, m j)`, and `|s + c| ≤ |S (j + 1)| + E j` (`chunk_step`). -/
def errAt (p q : Nat → ℚ) : Nat → ℚ
  | 0 => 0
  | j + 1 => (1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * errAt p q j +
      Rules.Num.rhoChunk8 * max |partSum p j| (chunkMax q j) +
      Rules.Num.sigmaChunk8 * |partSum p (j + 1)| + Rules.Num.alphaChunk8

/-- **The tolerance of `Post`** for row values `va`, column values `vb` and the reduction length
`K`: the error bound after the loop's `2·steps K` chunks,
`errAt (prodV va vb K) (liftV va vb K) (2 * steps K)`. It is explicit and computable from the
inputs, and depends on the order of the products (through the partial sums at each chunk of `32`)
and on each chunk's largest lifted product. It is never above `tolA K (dotLift va vb K)`
(`tol_le_tolA`). -/
def tol (va vb : Nat → F8E5) (K : Nat) : ℚ := errAt (prodV va vb K) (liftV va vb K) (2 * steps K)

/-- The bound in terms of one magnitude `L` (the sum of the lifted products, `dotLift`, will do):
`DotLoop.bound (ρ + σ) α n L = ((1 + ρ + σ)^n − 1)·L + n·(1 + ρ + σ)^n·α`, `n = 2·steps K`. `tol`
is never above it (`tol_le_tolA`), and equals it, up to the `α` terms, on an input whose only
product is the first, of normal operands. Its relative part `(1 + 2^-7 + 2^-12)^n − 1` is about `0.137` at `K = 512`
and `1.793` at `K = 4096`, below `1.8` on covered instances (`tolA_rel_lt`, `tolA_le`). -/
def tolA (K : Nat) (L : ℚ) : ℚ :=
  DotLoop.bound (Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) Rules.Num.alphaChunk8 (2 * steps K) L

theorem rho_eq : Rules.Num.rhoChunk8 = 1 / 128 := by unfold Rules.Num.rhoChunk8; norm_num

theorem sigma_eq : Rules.Num.sigmaChunk8 = 1 / 4096 := by unfold Rules.Num.sigmaChunk8; norm_num

theorem alpha_eq : Rules.Num.alphaChunk8 = 1 / 2 ^ 126 := by unfold Rules.Num.alphaChunk8; norm_num

/-- `1 + ρ + σ = 1 + 2^-7 + 2^-12 = 4129/4096`. -/
theorem q_eq : (1 : ℚ) + (Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) = 4129 / 4096 := by
  rw [rho_eq, sigma_eq]; norm_num

theorem errAt_succ (p q : Nat → ℚ) (j : Nat) :
    errAt p q (j + 1) = (1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * errAt p q j +
      Rules.Num.rhoChunk8 * max |partSum p j| (chunkMax q j) +
      Rules.Num.sigmaChunk8 * |partSum p (j + 1)| + Rules.Num.alphaChunk8 := rfl

theorem errAt_nonneg (p q : Nat → ℚ) : ∀ j, 0 ≤ errAt p q j
  | 0 => le_rfl
  | j + 1 => by
    have h := errAt_nonneg p q j
    have hm : 0 ≤ max |partSum p j| (chunkMax q j) := (abs_nonneg _).trans (le_max_left _ _)
    have hS := abs_nonneg (partSum p (j + 1))
    rw [errAt_succ, rho_eq, sigma_eq, alpha_eq]
    have : (0 : ℚ) < 1 / 2 ^ 126 := by norm_num
    linarith

theorem partSum_succ (p : Nat → ℚ) (j : Nat) :
    partSum p (j + 1) = partSum p j + ∑ k : Fin 32, p (32 * j + k) := by
  unfold partSum
  rw [show 32 * (j + 1) = 32 * j + 32 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => p (32 * j + k)) 32]

theorem le_chunkMax (q : Nat → ℚ) (j : Nat) (k : Fin 32) : q (32 * j + k) ≤ chunkMax q j :=
  Finset.le_sup' (fun k : Fin 32 => q (32 * j + k)) (Finset.mem_univ k)

theorem chunkMax_le {q : Nat → ℚ} {j : Nat} {M : ℚ} (h : ∀ k : Fin 32, q (32 * j + k) ≤ M) :
    chunkMax q j ≤ M :=
  Finset.sup'_le _ _ fun k _ => h k

/-- The error bound is never above the bound in terms of a magnitude `L` that bounds every
partial sum and every chunk's largest lifted product up to chunk `n`. -/
theorem errAt_le_bound (p q : Nat → ℚ) {L : ℚ} {n : Nat} (hS : ∀ j ≤ n, |partSum p j| ≤ L)
    (hq : ∀ j < n, chunkMax q j ≤ L) : ∀ j ≤ n,
    errAt p q j ≤ DotLoop.bound (Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) Rules.Num.alphaChunk8 j L
  | 0, _ => by simp [errAt, DotLoop.bound]
  | j + 1, hj => by
    have ih := errAt_le_bound p q hS hq j (by omega)
    have hL : 0 ≤ L := (abs_nonneg _).trans (hS 0 (Nat.zero_le _))
    have hm : max |partSum p j| (chunkMax q j) ≤ L := max_le (hS j (by omega)) (hq j (by omega))
    have hS1 := hS (j + 1) hj
    unfold DotLoop.bound at ih ⊢
    rw [q_eq, alpha_eq] at ih ⊢
    rw [errAt_succ, rho_eq, sigma_eq, alpha_eq, pow_succ (4129 / 4096 : ℚ) j]
    push_cast
    have hP1 : (1 : ℚ) ≤ (4129 / 4096) ^ j := one_le_pow₀ (by norm_num)
    have h1 := mul_le_mul_of_nonneg_left ih (show (0 : ℚ) ≤ 4129 / 4096 by norm_num)
    have h2 : (1 / 2 ^ 126 : ℚ) ≤ (4129 / 4096) ^ j * (4129 / 4096) * (1 / 2 ^ 126) := by
      have : (1 : ℚ) ≤ (4129 / 4096) ^ j * (4129 / 4096) := by linarith
      linarith
    linarith

/-- The relative part of `tolA` on covered instances (`K ≤ 4096`, `Inst.ok.K_le`): below `1.8`. -/
theorem tolA_rel_lt {K : Nat} (hK : K ≤ 2 ^ 12) :
    (1 + (Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8)) ^ (2 * steps K) - 1 < 9 / 5 := by
  have hT : 2 * steps K ≤ 128 := by unfold steps; omega
  rw [q_eq]
  have h2 := pow_le_pow_right₀ (show (1 : ℚ) ≤ 4129 / 4096 by norm_num) hT
  have h3 : (4129 / 4096 : ℚ) ^ 128 < 1 + 9 / 5 := by norm_num
  linarith

/-- On covered instances `tolA K L` is at most `1.8·L + 10^-35`. -/
theorem tolA_le {K : Nat} (hK : K ≤ 2 ^ 12) {L : ℚ} (hL : 0 ≤ L) :
    tolA K L ≤ 9 / 5 * L + 1 / 10 ^ 35 := by
  have hr := tolA_rel_lt hK
  have hT : 2 * steps K ≤ 128 := by unfold steps; omega
  unfold tolA DotLoop.bound
  rw [q_eq] at hr ⊢
  rw [alpha_eq]
  have h2 := pow_le_pow_right₀ (show (1 : ℚ) ≤ 4129 / 4096 by norm_num) hT
  have h3 : (4129 / 4096 : ℚ) ^ 128 < 3 := by norm_num
  have hT' : ((2 * steps K : Nat) : ℚ) ≤ 128 := by exact_mod_cast hT
  have hP : 0 ≤ (4129 / 4096 : ℚ) ^ (2 * steps K) := by positivity
  have e1 : ((4129 / 4096 : ℚ) ^ (2 * steps K) - 1) * L ≤ 9 / 5 * L :=
    mul_le_mul_of_nonneg_right hr.le hL
  have e2 : ((2 * steps K : Nat) : ℚ) * (4129 / 4096) ^ (2 * steps K) ≤ 128 * 3 :=
    mul_le_mul hT' (by linarith) hP (by norm_num)
  have e3 : ((2 * steps K : Nat) : ℚ) * (4129 / 4096) ^ (2 * steps K) * (1 / 2 ^ 126) ≤
      128 * 3 * (1 / 2 ^ 126) := mul_le_mul_of_nonneg_right e2 (by norm_num)
  have e4 : (128 * 3 * (1 / 2 ^ 126) : ℚ) ≤ 1 / 10 ^ 35 := by norm_num
  linarith

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F8E5) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The sum of the products' lifted magnitudes, `Σ liftMag (va k) · liftMag (vb k)` over `k < K`:
`Σ|va k · vb k|` when no operand is subnormal. -/
def dotLift (va vb : Nat → F8E5) (K : Nat) : ℚ :=
  ∑ k ∈ Finset.range K, Rules.Num.liftMag .f8e5 (va k).toRat * Rules.Num.liftMag .f8e5 (vb k).toRat

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite `float8e5` values `va k`, `vb k` (`k < K`) in the initial
memory, `c` is the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32`
value `v` within `E = tol va vb K` (the error bound over the loop's `2·cdiv(K, 64)` chunks of `32`
products, from the exact partial sums and each chunk's largest lifted product) of the exact sum
`S = Σ va k · vb k`; and when there is room in `float16`
(`|S| + E ≤ 65504`), `c` is finite and within `E` plus one `float16` rounding
(`(|S| + E)·2^-11 + 2^-25`) of `S`. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ va vb : Nat → F8E5,
      (∀ k, k < d.K → init.read .f8e5 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f8e5 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum va vb d.K| ≤ tol va vb d.K ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum va vb d.K| + tol va vb d.K ≤ Math.Float.Format.binary16.maxVal →
          c.isFinite = true ∧
          |c.toRat - dotSum va vb d.K| ≤ tol va vb d.K +
            (|dotSum va vb d.K| + tol va vb d.K) * Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta)

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
  /-- [rule] `Num.DotE5m2Chunk`: `tl.dot` of `float8e5` operands into `float32` with
  `maxImpreciseAcc = 2^30` (the default on sm_90), any `input_precision` (the model's line 300),
  chunk by chunk. For each output element whose row of `a`, column of `b` and accumulator have
  values (are finite), with room below the `float32` range (`mag + relDot8 K·mag + absDot8 K`,
  `mag = |acc| + Σ|a·b|`), the result has a value, and there are partial values from `acc` to the
  result, one per chunk of `32` consecutive `k`, in increasing order: each is within
  `2^-7·M + 2^-12·|s + c| + 2^-126` of the previous value `s` plus the chunk's exact sum `c`, for
  every `M` at least `|s|` and each of the chunk's lifted products `liftMag a_k · liftMag b_k` (a
  subnormal operand counts as `2^-14`). -/
  dot : Rules.Num.DotE5m2Chunk P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 300) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 }
  /-- [rule] `Num.CvtF32E5m2`: the conversion of a finite `float32` within `float8e5`'s range into
  `float8e5` with the default rounding is its rounding to nearest even, up to NaN payload; the
  model's `other=0.0` of lines 297–298, the `float32` `+0.0` converted to the pointers' dtype
  `float8e5`, is then `+0`. -/
  cvt : Rules.Num.CvtF32E5m2 P
  /-- [rule] `Num.CvtTotal`: the conversion from `float32` to `float8e5` with the default
  rounding (`other=0.0`, lines 297–298) always has a result. -/
  cvt_total : Rules.Num.CvtTotal P Math.Float.Format.binary32 Math.Float.Format.e5m2 .rn

end Kernels.MatmulE5m2
