import TVA

/-!
# `streamk_matmul_f16`: Stream-K matrix multiplication, `float16` into `float32`

The kernel is `streamk_matmul_kernel` of `source/streamk_matmul.py` (lines 26–64; it reads the
module-level names `triton`, its decorator, and `tl`, lines 22–23). Its host `streamk_matmul`
(lines 70–93) launches it on `float16` matrices `a` (`M × K`) and `b` (`K × N`) into a new
`float32` matrix `c = torch.zeros(M, N)`, with a 1-D grid of `num_programs` programs (the SM count
unless the caller passes one), `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`; `SETUP.md` lists the
test's launches.

Stream-K: the `num_tiles · iters_per_tile` iterations of the `K` loop of all tiles are split into
contiguous ranges of `iters_per_program`, one per program. A tile one program owns entirely is
stored; a tile split across programs is accumulated with `tl.atomic_add` by each program that owns
part of it.

What is claimed: every covered launch ends normally; every element `(i, j)` of `c` holds a
`float32` value, which, when row `i` of `a` and column `j` of `b` are finite, is finite and within
`tol = 2^-15 · tolMag + 2^-100` (stated chunk by chunk of the `K` loop, from the element's
operands) of the exact `Σ_k a[i,k]·b[k,j]`, whatever the split of its tile between programs and the
order the atomic additions ran in; every byte outside the matrix `c` is unchanged.

The conditions: `M`, `N`, `K` positive, `M ≤ 2^22`, `N` and `K` multiples of `128` at most `2^17`,
`K ≤ 2^13`; the strides and the `int32` index products, tile counts and iteration counts do not
wrap; `total_iters` and `iters_per_program` are the host's, and the programs' ranges cover every
iteration; the rows of `c` do not overlap; the buffers hold the matrices, the matrix `c` starts
zero and shares no byte with `a` or `b`. Every `float16` input is covered.

The tolerance rests on the device facts for `tl.dot` of `float16` operands into `float32`
(`Device.DotF16`), for `tl.atomic_add` on `float32` (`Device.AtomicAddF32`), and their totality
(`Device`).
-/

namespace Claims.StreamKMatmul

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
  /-- `M + 127 ≤ 2^31`: `M` is an `int32`, and the tile rows `pid_m * 128 + i` (line 48) stay
  below `cdiv(M, 128) * 128 ≤ M + 127`, so the `int32` row indices do not wrap. -/
  M_le : d.M + 127 ≤ 2 ^ 31
  /-- `N + 128 ≤ 2^31`: `N` is an `int32`, `tl.cdiv(N, 128)` computes `N + 127` in `int32`
  (line 37), and the tile columns stay below `N + 127` (line 49). -/
  N_le : d.N + 128 ≤ 2 ^ 31
  /-- `K ≤ 2^13`: at most `128` chunks, which the tolerance of `Post` relies on (README,
  Tolerances); it is then below `0.4%` of `A = Σ_k mag16 (a[i,k]) · mag16 (b[k,j])` (`Σ|a·b|` when
  no operand is subnormal) on every input.
  The test's largest `K` is `4097`. It also makes `K + 63 < 2^31` for `tl.cdiv(K, 64)` (line 38). -/
  K_le : d.K ≤ 2 ^ 13
  /-- `N` is a model dimension (the GEMM's output features): a multiple of 128, at most `2^17`. -/
  N_mul : d.N % 128 = 0
  N_le17 : d.N ≤ 2 ^ 17
  /-- `K` is a model dimension (the GEMM's inner features): a multiple of 128, at most `2^17`. -/
  K_mul : d.K % 128 = 0
  K_le17 : d.K ≤ 2 ^ 17
  /-- `M` counts tokens, a context length: at most `2^22`. -/
  M_le22 : d.M ≤ 2 ^ 22
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

/-! ## The specification

The tolerance follows the kernel's `K` loop: chunk `t < T = steps K` holds the products
`k ∈ [64·t, min(64·t + 64, K))`, one `tl.dot` of `K = 64` (the lanes past `K` are the loads'
zeros). It is stated per element, from the element's own operands. -/

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The magnitude by which the tensor core aligns a `float16` operand value `q`: `0` for `0`, and
otherwise `|q|`, or the smallest normal magnitude `2^-14` when `|q|` is below it (a subnormal). It is
`|q|` for zero and normal values. The library's `Device.liftMag .f16`, which `Device.DotF16`
counts its magnitudes by. -/
def mag16 (q : ℚ) : ℚ := if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ))

/-- The products of chunk `t`: `k ∈ [64·t, min(64·t + 64, K))`. -/
def chunk (K t : Nat) : Finset Nat := Finset.Ico (64 * t) (min (64 * t + 64) K)

/-- The exact sum of chunk `t`, `Σ_{k ∈ chunk t} va k · vb k`. -/
def chunkSum (va vb : Nat → F16) (K t : Nat) : ℚ :=
  ∑ k ∈ chunk K t, (va k).toRat * (vb k).toRat

/-- The lifted magnitude of chunk `t`, `Σ_{k ∈ chunk t} mag16 (va k) · mag16 (vb k)`: the
`Σ liftMag a · liftMag b` of `DotF16` for that `tl.dot`. -/
def chunkMag (va vb : Nat → F16) (K t : Nat) : ℚ :=
  ∑ k ∈ chunk K t, mag16 (va k).toRat * mag16 (vb k).toRat

/-- The exact sum of the first `s` chunks, `Σ_{u < s} chunkSum u` (`0` for `s = 0`). -/
def pre (va vb : Nat → F16) (K s : Nat) : ℚ := ∑ u ∈ Finset.range s, chunkSum va vb K u

/-- The largest exact sum of consecutive chunks ending before chunk `t`:
`max_{s ≤ t} |pre t − pre s|`. A program whose range starts at chunk `s ≤ t` holds, before its
`tl.dot` of chunk `t`, an accumulator of exact value `pre t − pre s`, so this bounds the exact value
of the accumulator at chunk `t` whatever the split of the tile between programs. -/
def dev (va vb : Nat → F16) (K t : Nat) : ℚ :=
  (Finset.range (t + 1)).sup' Finset.nonempty_range_add_one fun s => |pre va vb K t - pre va vb K s|

/-- **The magnitude the tolerance scales with**: `Σ_{t ≤ T} dev t + Σ_{t < T} chunkMag t`,
`T = steps K`: the largest exact values a program's accumulator can hold at the `T + 1` chunk
boundaries, whatever the split of the tile between programs, plus the products' lifted magnitudes.
It is what the `tl.dot`s scale their errors by (`DotF16`: `|acc|` plus the chunk's magnitude), and it
bounds the magnitudes of the partial sums the atomic additions add up. -/
def tolMag (va vb : Nat → F16) (K : Nat) : ℚ :=
  (∑ t ∈ Finset.range (steps K + 1), dev va vb K t) +
    ∑ t ∈ Finset.range (steps K), chunkMag va vb K t

/-- **The tolerance of an element**: `2^-15 · tolMag + 2^-100`, whatever the split of its tile
between programs and the order of the atomic additions. -/
def tol (va vb : Nat → F16) (K : Nat) : ℚ := 2 ^ (-15 : ℤ) * tolMag va vb K + 2 ^ (-100 : ℤ)

/-- What the launch leaves in element `(i, j)` of `c`: some `float32` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is
finite and within `tol va vb K` of the exact sum `S = Σ va k · vb k`, whatever order the atomic
additions ran in. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F32, fin.read .f32 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      c.isFinite = true ∧ |c.toRat - dotSum va vb d.K| ≤ tol va vb d.K

/-- What the launch leaves: every element of the matrix `c` as `OutSpec` says; every byte outside
the matrix `c` unchanged (`a`, `b`, and the elements of `c`'s storage outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i j, i < d.M → j < d.N → OutSpec d init fin i j) ∧
    ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `streamk_matmul_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The host's launch `streamk_matmul(a, b, num_programs)` on contiguous `a` (`M × K`, strides
`(K, 1)`) and `b` (`K × N`, strides `(N, 1)`) and the fresh `c` (`M × N`, strides `(N, 1)`), with
`np` programs, `total_iters = cdiv(M, 128) · cdiv(N, 128) · cdiv(K, 64)` and
`iters_per_program = cdiv(total_iters, np)` (lines 79–81); `a`, `b`, `c` at `2^36`, `2·2^36`,
`3·2^36`. -/
def mk (M N K np : Nat) : Inst where
  M := M
  N := N
  K := K
  stride_am := K
  stride_ak := 1
  stride_bk := N
  stride_bn := 1
  stride_cm := N
  stride_cn := 1
  num_programs := np
  iters_per_program := ((M + 127) / 128 * ((N + 127) / 128) * ((K + 63) / 64) + np - 1) / np
  total_iters := (M + 127) / 128 * ((N + 127) / 128) * ((K + 63) / 64)
  a := ⟨2 ^ 36, M * K⟩
  b := ⟨2 * 2 ^ 36, K * N⟩
  c := ⟨3 * 2 ^ 36, M * N⟩

theorem idx_lt {M K i k : Nat} (hi : i < M) (hk : k < K) : i * K + k * 1 < M * K := by
  have : (i + 1) * K ≤ M * K := Nat.mul_le_mul_right _ hi
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

theorem cdiv_cover {t np : Nat} (hnp : 0 < np) : t ≤ np * ((t + np - 1) / np) := by
  have h1 := Nat.div_add_mod (t + np - 1) np
  have h2 := Nat.mod_lt (t + np - 1) hnp
  omega

/-- Every such launch is covered when the shapes are positive, the `int32` bounds hold, `K ≤ 2^13`,
there is a program and `num_programs · iters_per_program < 2^31`, each matrix fits in `2^36`
bytes, `N` and `K` are multiples of 128 up to `2^17`, and `M ≤ 2^22`. -/
theorem mk_ok {M N K np : Nat} (hM : 0 < M) (hN : 0 < N) (hK : 0 < K) (hMle : M + 127 ≤ 2 ^ 31)
    (hNle : N + 128 ≤ 2 ^ 31) (hKle : K ≤ 2 ^ 13) (ham : (M - 1) * K < 2 ^ 31)
    (hbk : (K - 1) * N < 2 ^ 31) (hcm : (M - 1) * N < 2 ^ 31) (hnp : 0 < np)
    (hprog : np * (((M + 127) / 128 * ((N + 127) / 128) * ((K + 63) / 64) + np - 1) / np) < 2 ^ 31)
    (hsa : M * K * 2 ≤ 2 ^ 36) (hsb : K * N * 2 ≤ 2 ^ 36) (hsc : M * N * 4 ≤ 2 ^ 36)
    (hNm : N % 128 = 0) (hN17 : N ≤ 2 ^ 17) (hKm : K % 128 = 0) (hK17 : K ≤ 2 ^ 17)
    (hM22 : M ≤ 2 ^ 22) :
    (mk M N K np).ok := by
  refine ⟨hM, hN, hK, hMle, hNle, hKle, hNm, hN17, hKm, hK17, hM22, by simp only [mk]; omega, by simp only [mk]; omega,
    by simp only [mk]; omega, by simp only [mk]; omega, by simp only [mk]; omega,
    by simp only [mk]; omega, ham, by simp only [mk]; omega, hbk, by simp only [mk]; omega, hcm,
    by simp only [mk]; omega, rfl, cdiv_cover hnp, hprog,
    fun i k hi hk => idx_lt hi hk, fun k j hk hj => idx_lt hk hj, fun i j hi hj => idx_lt hi hj,
    ?_, ?_⟩
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, mk, Set.mem_ofPred_eq, DType.bytes] at hx hy
      have : ((M * N : Nat) : Int) * 4 ≤ 2 ^ 36 := by exact_mod_cast hsc
      have : ((M * K : Nat) : Int) * 2 ≤ 2 ^ 36 := by exact_mod_cast hsa
      push_cast at hx hy this
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩; exact idx_lt hi hj
    · rintro _ ⟨i, k, hi, hk, rfl⟩; exact idx_lt hi hk
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, mk, Set.mem_ofPred_eq, DType.bytes] at hx hy
      have : ((M * N : Nat) : Int) * 4 ≤ 2 ^ 36 := by exact_mod_cast hsc
      have : ((K * N : Nat) : Int) * 2 ≤ 2 ^ 36 := by exact_mod_cast hsb
      push_cast at hx hy this
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩; exact idx_lt hi hj
    · rintro _ ⟨k, j, hk, hj, rfl⟩; exact idx_lt hk hj

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

/-- Every aligned `float32` read of `testMem` gives `+0`. -/
theorem testMem_read {a : Int} (ha : Mem.Aligned .f32 a) : testMem.read .f32 a = some (FloatV.ofPat 0) := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by
    simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem claim_satisfiable : claim.Satisfiable := by
  have hvalid : ∀ {dd : DType} (b : Buf dd), 0 ≤ b.base → b.base + b.len * dd.bytes ≤ 2 ^ 64 →
      b.base % dd.bytes = 0 → b.Valid testMem :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨mk 512 512 512 132, testMem, mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num),
    hvalid _ (by norm_num [mk]) (by norm_num [mk, DType.bytes]) (by norm_num [mk, DType.bytes]),
    hvalid _ (by norm_num [mk]) (by norm_num [mk, DType.bytes]) (by norm_num [mk, DType.bytes]),
    hvalid _ (by norm_num [mk]) (by norm_num [mk, DType.bytes]) (by norm_num [mk, DType.bytes]),
    fun i j hi hj => ?_⟩
  have h1 : (FloatV.ofPat 0 : F32).isFinite = true := by decide
  have h2 : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel
  refine ⟨FloatV.ofPat 0, testMem_read ⟨?_, ?_, ?_⟩, h1, h2⟩ <;>
    simp only [Buf.ptr, mk, DType.bytes] at hi hj ⊢ <;> omega

/-! ## Coverage of the test's launches (`SETUP.md`)

`(M, N, K, num_programs)` of the test (lines 99–106), with the H200's 132 SMs for
`num_programs=None`, whose `N` and `K` are multiples of 128 (README, Coverage). -/
def setup_512_512_512_p132 : Inst := mk 512 512 512 132
theorem setup_512_512_512_p132_ok : setup_512_512_512_p132.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num)
def setup_3072_4096_1024_p132 : Inst := mk 3072 4096 1024 132
theorem setup_3072_4096_1024_p132_ok : setup_3072_4096_1024_p132.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num)
def setup_8192_8192_4096_p132 : Inst := mk 8192 8192 4096 132
theorem setup_8192_8192_4096_p132_ok : setup_8192_8192_4096_p132.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num)

end Claims.StreamKMatmul
