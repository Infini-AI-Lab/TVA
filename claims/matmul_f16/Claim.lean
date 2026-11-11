import TVA

/-!
# `matmul_f16`: matrix multiplication, Triton tutorial 03

The kernel is `matmul_kernel` of `source/03-matrix-multiplication.py` (lines 228–316; it reads the
module-level names `triton`, its decorator, and `tl`, bound by the imports of lines 154–155), with
the config of `SETUP.md`. Its host function `matmul` (lines 330–350) launches it on `float16`
matrices `a` (`M × K`) and `b` (`K × N`) into a new `float16` matrix `c` (`M × N`): the test
(`512 × 512 × 512`, lines 359–370) and the benchmark's `fp16` line (square sizes
`256, 384, …, 4096`, lines 400–440); `SETUP.md` lists them. Each program computes one `128 × 256`
tile of `c` by a loop of `tl.dot`s over `K` in steps of `64`.

What is claimed: every covered launch ends normally; every element `c[i, j]` holds a `float16`
value; when row `i` of `a` and column `j` of `b` are finite, `c[i, j]` is the `float16` rounding of a
finite `float32` value within `tol = 2^-15·pathMag + 2^-100` of the exact dot product `S`, and when
`|S| + tol ≤ 65504` it is finite and within `2^-11·|S| + 2^-15·pathMag + 2^-24` of `S`; every byte
outside the matrix `c` is unchanged.

The conditions: the strides are positive (the kernel's `tl.assume`s) and below `2^31`; the `int32`
sizes, offsets and the grid do not wrap; `K ≤ 2^13`; `N` and `K` are positive multiples of `128`, at
most `2^17`, and `M ≤ 2^22`; the buffers hold the matrices; distinct elements of `c` are distinct in
storage, and `c` shares no byte with `a` or `b`. Every `float16` input is covered.

The tolerance rests on the device fact for a `float16` `tl.dot` into `float32` (`Device.DotF16`).
-/

namespace Claims.Matmul

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

/-- The instances covered. Shapes, strides and layouts are variables; the bounds keep the kernel's
`int32` arithmetic from wrapping, the grid within CUDA's limit, the tolerance meaningful
(`K ≤ 2^13`), and the elements of `c` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and `tl.cdiv(M, 128)` computes `M + 127` in `int32`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 256 ≤ 2^31`: likewise for `tl.cdiv(N, 256)`. -/
  N_le : d.N + 256 ≤ 2 ^ 31
  /-- The grid, `cdiv(M, 128) · cdiv(N, 256)` programs, has fewer than `2^31` programs (CUDA's limit
  on a 1-D grid, `Launch.Valid`; `pid` is an `int32`). -/
  grid_lt : (d.M + 127) / 128 * ((d.N + 255) / 256) < 2 ^ 31
  /-- `K ≤ 2^13`: the loop has at most `128` steps, so the error the steps compound stays within
  the tolerance of `Post` (README, Tolerances). The host's largest `K` is `4096`. -/
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
  /-- The `int32` offsets of `a_ptrs` (line 285), `i * stride_am + k * stride_ak` for `i < M` (the
  rows are taken `% M`, line 282) and `k < 64`, and the step `64 * stride_ak` (line 302), do not
  wrap. -/
  a_fit : (d.M - 1) * d.stride_am + 64 * d.stride_ak < 2 ^ 31
  /-- The `int32` offsets of `b_ptrs` (line 286), `k * stride_bk + j * stride_bn` for `k < 64` and
  `j < N` (`% N`, line 283), and the step `64 * stride_bk` (line 303), do not wrap. -/
  b_fit : 64 * d.stride_bk + (d.N - 1) * d.stride_bn < 2 ^ 31
  /-- The `int32` products `stride_cm * offs_cm` of line 314 do not wrap on the rows below `M`. -/
  cm_fit : (d.M - 1) * d.stride_cm < 2 ^ 31
  /-- The `int32` products `stride_cn * offs_cn` of line 314 do not wrap on the columns below `N`. -/
  cn_fit : (d.N - 1) * d.stride_cn < 2 ^ 31
  /-- Distinct elements of the matrix `c` are distinct elements of its storage (any layout in which
  they do not overlap: row-major, padded, column-major, …): otherwise two lanes or two programs
  store to one element. -/
  c_inj : ∀ i j i' j', i < d.M → j < d.N → i' < d.M → j' < d.N →
    i * d.stride_cm + j * d.stride_cn = i' * d.stride_cm + j' * d.stride_cn → i = i' ∧ j = j'
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
  /-- `M` counts tokens (the rows of the activations `a`): a context length, at most `2^22`. -/
  M_ctx : d.M ≤ 2 ^ 22
  /-- `N` is a model dimension (the output features): a positive multiple of `128`, at most
  `2^17`. Every column tile of `256` is then full or half full (`128 ∣ N`). -/
  N_dim : 0 < d.N ∧ d.N % 128 = 0 ∧ d.N ≤ 2 ^ 17
  /-- `K` is a model dimension (the input features): a positive multiple of `128`, at most `2^17`.
  Every `k` step of `64` is then full (`64 ∣ K`): the masks `k < K - 64·t` of lines 297–298 hold on
  every lane. -/
  K_dim : 0 < d.K ∧ d.K % 128 = 0 ∧ d.K ≤ 2 ^ 17

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

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- Product `e` of the reduction: `va e · vb e` for `e < K`, and `0` from `K` on (the lanes the
masks of lines 297–298 disable, loaded as `+0`). -/
def prodV (va vb : Nat → F16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The lifted magnitude of product `e`: `liftMag (va e) · liftMag (vb e)` for `e < K`, `0` from `K`
on. `Device.liftMag .f16 q` is `|q|` for zero and normal values and `2^-14` (the smallest normal
magnitude) for a nonzero subnormal one: the magnitude by which the tensor core aligns an operand,
which `Device.DotF16` counts by. -/
def liftV (va vb : Nat → F16) (K e : Nat) : ℚ :=
  if e < K then Device.liftMag .f16 (va e).toRat * Device.liftMag .f16 (vb e).toRat else 0

/-- The exact partial sum before step `t` of a sequence of products `p`: `p e` for `e < 64·t`. -/
def partSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), p e

/-- The lifted magnitude of step `t`: `Σ q e` for `64·t ≤ e < 64·t + 64`. -/
def stepMag (q : Nat → ℚ) (t : Nat) : ℚ := ∑ k : Fin 64, q (64 * t + k)

/-- **The magnitude the tolerance scales with**, for row values `va`, column values `vb` and the
reduction length `K`: `Σ_{t < T} (|S t| + L t)` over the loop's `T = steps K` steps, with
`S t = partSum (prodV va vb K) t` the exact partial sum before step `t` (the products `e < 64·t`) and
`L t = stepMag (liftV va vb K) t` the lifted magnitude of step `t`'s `64` products. It is what each
step's `tl.dot` scales its error by (`Device.DotF16`: the accumulator's magnitude, close to
`|S t|`, plus the step's products), summed over the steps; it is at least `|Σ va k · vb k|`. -/
def pathMag (va vb : Nat → F16) (K : Nat) : ℚ :=
  ∑ t ∈ Finset.range (steps K), (|partSum (prodV va vb K) t| + stepMag (liftV va vb K) t)

/-- **The tolerance of `Post`** for row values `va`, column values `vb` and the reduction length
`K`: `2^-15 · pathMag va vb K + 2^-100`. It is explicit and computable from the inputs. -/
def tol (va vb : Nat → F16) (K : Nat) : ℚ := 2 ^ (-15 : ℤ) * pathMag va vb K + 2 ^ (-100 : ℤ)

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is
the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` within
`E = tol va vb K` of the exact sum `S = Σ va k · vb k`; and when there is room in `float16`
(`|S| + E ≤ 65504`), `c` is finite and within `2^-11·|S| + 2^-15·M + 2^-24` of `S`
(`M = pathMag va vb K`: the accumulation's tolerance and one `float16` rounding). -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum va vb d.K| ≤ tol va vb d.K ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum va vb d.K| + tol va vb d.K ≤ Format.binary16.maxVal →
          c.isFinite = true ∧
          |c.toRat - dotSum va vb d.K| ≤
            2 ^ (-11 : ℤ) * |dotSum va vb d.K| + 2 ^ (-15 : ℤ) * pathMag va vb d.K + 2 ^ (-24 : ℤ))

/-- What the launch leaves: every element of the matrix `c` as `OutSpec` says; every byte outside
the matrix `c` unchanged (`a`, `b`, and the elements of `c`'s storage outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i j, i < d.M → j < d.N → OutSpec d init fin i j) ∧
    ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `matmul_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Coverage: the launches of `SETUP.md`

`hostInst M N K` is the instance `matmul(a, b)` builds for a contiguous `float16` `a` (`M × K`,
strides `(K, 1)`), a contiguous `b` (`K × N`, strides `(N, 1)`) and the fresh `c` it allocates
(`M × N`, strides `(N, 1)`). The three buffers sit at fixed, aligned, far-apart addresses; the claim
covers any addresses `Inst.ok` allows, and these only witness the sizes. -/

/-- The host's instance for the sizes `(M, N, K)`. -/
def hostInst (M N K : Nat) : Inst where
  M := M
  N := N
  K := K
  stride_am := K
  stride_ak := 1
  stride_bk := N
  stride_bn := 1
  stride_cm := N
  stride_cn := 1
  a := ⟨2 ^ 32, M * K⟩
  b := ⟨2 ^ 34, K * N⟩
  c := ⟨2 ^ 36, M * N⟩

/-- `t·K + k < M·K` for `t < M`, `k < K`. -/
theorem row_lt {t M K k : Nat} (ht : t < M) (hk : k < K) : t * K + k < M * K := by
  have h : (t + 1) * K ≤ M * K := Nat.mul_le_mul_right K ht
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Row-major indices of distinct elements differ. -/
theorem rm_inj {N i j i' j' : Nat} (hj : j < N) (hj' : j' < N) (h : i * N + j = i' * N + j') :
    i = i' ∧ j = j' := by
  have hi : i = i' := by
    rcases Nat.lt_trichotomy i i' with hlt | heq | hgt
    · have h1 : (i + 1) * N ≤ i' * N := Nat.mul_le_mul_right N hlt
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega
    · exact heq
    · have h1 : (i' + 1) * N ≤ i * N := Nat.mul_le_mul_right N hgt
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega
  subst hi
  exact ⟨rfl, by omega⟩

/-- Two buffers whose byte ranges are in order share no byte. -/
theorem bytes_disjoint {d₁ d₂ : DType} {x : Buf d₁} {y : Buf d₂}
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The host's instance is covered when its sizes meet `Inst.ok`'s bounds and fit the address
layout of `hostInst`. -/
theorem hostInst_ok {M N K : Nat}
    (h : 0 < N ∧ 0 < K ∧ M + 128 ≤ 2 ^ 31 ∧ N + 256 ≤ 2 ^ 31 ∧
      (M + 127) / 128 * ((N + 255) / 256) < 2 ^ 31 ∧ K ≤ 2 ^ 13 ∧
      (M - 1) * K + 64 < 2 ^ 31 ∧ 64 * N + (N - 1) < 2 ^ 31 ∧ (M - 1) * N < 2 ^ 31 ∧
      M * K ≤ 2 ^ 31 ∧ K * N ≤ 2 ^ 31 ∧ M * N ≤ 2 ^ 31 ∧
      M ≤ 2 ^ 22 ∧ N % 128 = 0 ∧ N ≤ 2 ^ 17 ∧ K % 128 = 0 ∧ K ≤ 2 ^ 17) :
    (hostInst M N K).ok := by
  obtain ⟨hN, hK, h1, h2, h3, h4, h5, h6, h7, hA, hB, hC, hM22, hN128, hN17, hK128, hK17⟩ := h
  have hal : ∀ i k, i < M → k < K → i * K + k * 1 < M * K := fun i k hi hk => by
    rw [Nat.mul_one]; exact row_lt hi hk
  have hbl : ∀ k j, k < K → j < N → k * N + j * 1 < K * N := fun k j hk hj => by
    rw [Nat.mul_one]; exact row_lt hk hj
  have hcl : ∀ i j, i < M → j < N → i * N + j * 1 < M * N := fun i j hi hj => by
    rw [Nat.mul_one]; exact row_lt hi hj
  have hcS : (hostInst M N K).c.elems (cIdx (hostInst M N K)) ⊆ (hostInst M N K).c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; exact hcl i j hi hj)
  have haS : (hostInst M N K).a.elems (aIdx (hostInst M N K)) ⊆ (hostInst M N K).a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; exact hal i k hi hk)
  have hbS : (hostInst M N K).b.elems (bIdx (hostInst M N K)) ⊆ (hostInst M N K).b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; exact hbl k j hk hj)
  refine ⟨h1, h2, h3, h4, ⟨hK, by simp only [hostInst]; omega⟩, ⟨by simp [hostInst], by simp [hostInst]⟩,
    ⟨hN, by simp only [hostInst]; omega⟩, ⟨by simp [hostInst], by simp [hostInst]⟩,
    ⟨hN, by simp only [hostInst]; omega⟩, ⟨by simp [hostInst], by simp [hostInst]⟩,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega, by simp only [hostInst]; omega,
    by simp only [hostInst]; omega,
    fun i j i' j' _ hj _ hj' he => rm_inj hj hj' (by simpa only [hostInst, Nat.mul_one] using he),
    hal, hbl, hcl, ?_, ?_, hM22, ⟨hN, hN128, hN17⟩, ⟨hK, hK128, hK17⟩⟩
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega

/-- The source's test (lines 359–370): `M = N = K = 512`. -/
def setup_test_512 : Inst := hostInst 512 512 512
theorem setup_test_512_ok : setup_test_512.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 256`. -/
def setup_bench_256 : Inst := hostInst 256 256 256
theorem setup_bench_256_ok : setup_bench_256.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 384`. -/
def setup_bench_384 : Inst := hostInst 384 384 384
theorem setup_bench_384_ok : setup_bench_384.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 512`. -/
def setup_bench_512 : Inst := hostInst 512 512 512
theorem setup_bench_512_ok : setup_bench_512.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 640`. -/
def setup_bench_640 : Inst := hostInst 640 640 640
theorem setup_bench_640_ok : setup_bench_640.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 768`. -/
def setup_bench_768 : Inst := hostInst 768 768 768
theorem setup_bench_768_ok : setup_bench_768.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 896`. -/
def setup_bench_896 : Inst := hostInst 896 896 896
theorem setup_bench_896_ok : setup_bench_896.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1024`. -/
def setup_bench_1024 : Inst := hostInst 1024 1024 1024
theorem setup_bench_1024_ok : setup_bench_1024.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1152`. -/
def setup_bench_1152 : Inst := hostInst 1152 1152 1152
theorem setup_bench_1152_ok : setup_bench_1152.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1280`. -/
def setup_bench_1280 : Inst := hostInst 1280 1280 1280
theorem setup_bench_1280_ok : setup_bench_1280.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1408`. -/
def setup_bench_1408 : Inst := hostInst 1408 1408 1408
theorem setup_bench_1408_ok : setup_bench_1408.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1536`. -/
def setup_bench_1536 : Inst := hostInst 1536 1536 1536
theorem setup_bench_1536_ok : setup_bench_1536.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1664`. -/
def setup_bench_1664 : Inst := hostInst 1664 1664 1664
theorem setup_bench_1664_ok : setup_bench_1664.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1792`. -/
def setup_bench_1792 : Inst := hostInst 1792 1792 1792
theorem setup_bench_1792_ok : setup_bench_1792.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 1920`. -/
def setup_bench_1920 : Inst := hostInst 1920 1920 1920
theorem setup_bench_1920_ok : setup_bench_1920.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2048`. -/
def setup_bench_2048 : Inst := hostInst 2048 2048 2048
theorem setup_bench_2048_ok : setup_bench_2048.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2176`. -/
def setup_bench_2176 : Inst := hostInst 2176 2176 2176
theorem setup_bench_2176_ok : setup_bench_2176.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2304`. -/
def setup_bench_2304 : Inst := hostInst 2304 2304 2304
theorem setup_bench_2304_ok : setup_bench_2304.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2432`. -/
def setup_bench_2432 : Inst := hostInst 2432 2432 2432
theorem setup_bench_2432_ok : setup_bench_2432.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2560`. -/
def setup_bench_2560 : Inst := hostInst 2560 2560 2560
theorem setup_bench_2560_ok : setup_bench_2560.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2688`. -/
def setup_bench_2688 : Inst := hostInst 2688 2688 2688
theorem setup_bench_2688_ok : setup_bench_2688.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2816`. -/
def setup_bench_2816 : Inst := hostInst 2816 2816 2816
theorem setup_bench_2816_ok : setup_bench_2816.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 2944`. -/
def setup_bench_2944 : Inst := hostInst 2944 2944 2944
theorem setup_bench_2944_ok : setup_bench_2944.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3072`. -/
def setup_bench_3072 : Inst := hostInst 3072 3072 3072
theorem setup_bench_3072_ok : setup_bench_3072.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3200`. -/
def setup_bench_3200 : Inst := hostInst 3200 3200 3200
theorem setup_bench_3200_ok : setup_bench_3200.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3328`. -/
def setup_bench_3328 : Inst := hostInst 3328 3328 3328
theorem setup_bench_3328_ok : setup_bench_3328.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3456`. -/
def setup_bench_3456 : Inst := hostInst 3456 3456 3456
theorem setup_bench_3456_ok : setup_bench_3456.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3584`. -/
def setup_bench_3584 : Inst := hostInst 3584 3584 3584
theorem setup_bench_3584_ok : setup_bench_3584.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3712`. -/
def setup_bench_3712 : Inst := hostInst 3712 3712 3712
theorem setup_bench_3712_ok : setup_bench_3712.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3840`. -/
def setup_bench_3840 : Inst := hostInst 3840 3840 3840
theorem setup_bench_3840_ok : setup_bench_3840.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 3968`. -/
def setup_bench_3968 : Inst := hostInst 3968 3968 3968
theorem setup_bench_3968_ok : setup_bench_3968.ok := hostInst_ok (by norm_num)

/-- The benchmark's `fp16` line (lines 400–440): `M = N = K = 4096`. -/
def setup_bench_4096 : Inst := hostInst 4096 4096 4096
theorem setup_bench_4096_ok : setup_bench_4096.ok := hostInst_ok (by norm_num)

/-! ## Not vacuous: a covered instance and memory -/

/-- The all-zero memory: every byte present and `0`. -/
def zeroMem : Mem := fun _ => some (0 : TVA.Bytes.Byte)

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_test_512, zeroMem, setup_test_512_ok, ?_, ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [setup_test_512, hostInst], by norm_num [setup_test_512, hostInst, DType.bytes],
      by norm_num [setup_test_512, hostInst, DType.bytes], fun _ _ => by simp [zeroMem]⟩

end Claims.Matmul
