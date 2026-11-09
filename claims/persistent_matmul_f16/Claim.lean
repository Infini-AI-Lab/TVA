import TVA

/-!
# `persistent_matmul_f16`: the prepared claim

Source: `source/09-persistent-matmul.py` (tutorial 09, with the config of `SETUP.md`). The kernel is
`matmul_kernel_persistent` (lines 301–357) with its `@triton.jit` helper `_compute_pid`
(lines 287–294); it reads the module-level names `triton` (its decorator) and `tl` (lines 40–41).
Its host function `matmul_persistent` (lines 360–380) launches `min(NUM_SMS, tiles)` programs, each
looping over the `128 × 256` tiles `start_pid, start_pid + NUM_SMS, …` of `c = a · b`, on `float16`
matrices `a` (`M × K`) and `b` (`K × N`) into a new `float16` matrix `c` (`M × N`). The launches of
`SETUP.md` are the file's test and benchmark at `M = N = 8192`, `K = 128, 256, …, 1024`.

This file holds the model (`compute_pid`, `matmul_kernel_persistent`, `model`, `launch`), the
conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`), the assumptions
(`Assumptions`) and the claim `Statement`, which it does not prove. It proves `consistent`,
`satisfiable`, `tol_rel_lt`, and that each launch of `SETUP.md` is covered (`setup_*_ok`).
-/

namespace Claims.PersistentMatmul

open TVA

/-! ## The model -/

/-- The helper `_compute_pid` (source lines 287–294), inlined at each call (lines 327 and 348).
`tile_id`, `num_pid_in_group` and `num_pid_m` are the `int32` scalars the calls pass;
`GROUP_SIZE_M` is the kernel's constexpr, passed directly, so it stays a compile-time value in the
body. The helper's last parameter `NUM_SMS` (the kernel's constexpr) is not used by its body, and
the model has no parameter for it. -/
def compute_pid (tile_id num_pid_in_group num_pid_m : I32) (GROUP_SIZE_M : Nat) : Prog (I32 × I32) := do
  -- line 289: `int32` `//` (`tl.idiv`)
  let group_id := (← tl.idiv (Tensor.scalar tile_id) (Tensor.scalar num_pid_in_group)).item
  -- line 290: `int32` scalar times the constexpr, built in `int32`; wrapping `*`
  let first_pid_m := group_id * (IntV.ofNat GROUP_SIZE_M : I32)
  -- line 291: Python's `min` with a tensor argument is `tl.minimum`, which makes the constexpr an
  -- `int32` tensor of its own type; on `int32` it is the exact `tl.iminimum`
  let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
    (Tensor.scalar (IntV.ofNat GROUP_SIZE_M : I32))).item
  -- line 292: `int32` `%` (`tl.irem`), then the wrapping `+`
  let pid_m := first_pid_m + (← tl.irem (Tensor.scalar tile_id) (Tensor.scalar group_size_m)).item
  -- line 293: `int32` `%`, then `int32` `//`
  let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar tile_id)
    (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
  -- line 294: the pair, which the call sites unpack
  return (pid_m, pid_n)

/-- `matmul_kernel_persistent` (source lines 301–357). The compile-time condition
`c_ptr.dtype.element_ty == tl.float8e4nv` of line 353 is false (`c_ptr` is `*fp16` on every
covered launch): the front end does not visit line 354 and takes the `else` branch, line 356.

The pointers are `*fp16` (the host passes `float16` tensors). `M`, `N`, `K` and the six strides are
`int32`: an integer argument in `[-2^31, 2^31)` other than `1` is a runtime `int32`, and each is
below `2^31` on covered instances (`Inst.ok`); at `1` an argument is the compile-time constant `1`,
which gives the same model (see `model`). The constexprs `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`BLOCK_SIZE_K`, `GROUP_SIZE_M`, `NUM_SMS` are Lean parameters. -/
def matmul_kernel_persistent (a_ptr b_ptr c_ptr : Ptr .f16) (M N K : I32)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) : Kernel where
  body ctx := do
    -- line 313: axis 0 of the 1-D grid, an `int32`
    let start_pid := tl.program_id ctx 0
    -- line 314: `tl.cdiv(M, BLOCK_SIZE_M)` is `(M + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: the
    -- constexpr `BLOCK_SIZE_M - 1` is computed by Python, then built in `int32` (an operand of `+`
    -- with the `int32` `M`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_m := (← tl.idiv (Tensor.scalar (M + (IntV.ofNat (BLOCK_SIZE_M - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_M : I32))).item
    -- line 315: the same with `N`, `BLOCK_SIZE_N`
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 316: the same with `K`, `BLOCK_SIZE_K`
    let k_tiles := (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
    -- line 317: wrapping `int32` `*`
    let num_tiles := num_pid_m * num_pid_n
    -- line 321: the `int32` `start_pid` minus the constexpr, built in `int32`; wrapping `-`
    let tile_id_c := start_pid - (IntV.ofNat NUM_SMS : I32)
    -- line 323
    let offs_k_for_mask : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K
    -- line 324: the constexpr times the `int32` scalar, built in `int32`; wrapping `*`
    let num_pid_in_group := (IntV.ofNat GROUP_SIZE_M : I32) * num_pid_n
    -- line 326: `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`: `int32` bounds, the
    -- constexpr step built in `int32`, an `int32` induction variable; `flatten` is a scheduling
    -- keyword (no meaning). The loop carries `tile_id_c`, the only name defined before it that its
    -- body assigns (line 347); nothing after the loop uses it
    let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat NUM_SMS : I32) tile_id_c
        fun tile_id tile_id_c => do
      -- line 327: `_compute_pid` inlined, its tuple unpacked
      let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m GROUP_SIZE_M
      -- lines 328–329: `int32` scalar times the constexpr
      let start_m := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)
      let start_n := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)
      -- lines 330–331: the scalar added to every lane of `tl.arange`, `int32`
      let offs_am : Tensor [BLOCK_SIZE_M] I32 := start_m + tl.arange 0 BLOCK_SIZE_M
      let offs_bn : Tensor [BLOCK_SIZE_N] I32 := start_n + tl.arange 0 BLOCK_SIZE_N
      -- line 332: `offs_am < M` (signed `int32`, the scalar broadcast); the Python `0` is built in
      -- `offs_am`'s dtype `int32`
      let offs_am := tl.where (offs_am.ltS M) offs_am (tl.full [BLOCK_SIZE_M] (IntV.ofInt 0 : I32))
      -- line 333: the same with `offs_bn`, `N`
      let offs_bn := tl.where (offs_bn.ltS N) offs_bn (tl.full [BLOCK_SIZE_N] (IntV.ofInt 0 : I32))
      -- line 334: `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)`: both
      -- hints on the value the `tl.where` of line 332 defines (in the body: checked each
      -- iteration), the pair's promise (groups of `BLOCK_SIZE_M` consecutive values,
      -- `BLOCK_SIZE_M` dividing the first of each group)
      let offs_am ← tl.max_contiguous_multiple_of offs_am BLOCK_SIZE_M BLOCK_SIZE_M
      -- line 335: the same for `offs_bn`, `BLOCK_SIZE_N`
      let offs_bn ← tl.max_contiguous_multiple_of offs_bn BLOCK_SIZE_N BLOCK_SIZE_N
      -- line 337: `+0.0` in every element
      let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
      -- line 338: `range(k_tiles)`: one argument, the name `k_tiles` evaluated twice (no
      -- operation); `int32` bounds `0`, `k_tiles` and step `1`, an `int32` induction variable; the
      -- loop carries `accumulator` (the only name defined before it that its body assigns)
      let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
          fun ki accumulator => do
        -- line 339: `int32` `ki * BLOCK_SIZE_K` added to every lane of `tl.arange`
        let offs_k : Tensor [BLOCK_SIZE_K] I32 := ki * (IntV.ofNat BLOCK_SIZE_K : I32) + tl.arange 0 BLOCK_SIZE_K
        -- line 340: `offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak` in `int32`
        -- (wrapping), broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`, then added to the `*fp16` pointer
        let a_ptrs := a_ptr + (offs_am.col BLOCK_SIZE_K * stride_am + offs_k.row BLOCK_SIZE_M * stride_ak)
        -- line 341: the same for `b`, `[BLOCK_SIZE_K, BLOCK_SIZE_N]`
        let b_ptrs := b_ptr + (offs_k.col BLOCK_SIZE_N * stride_bk + offs_bn.row BLOCK_SIZE_K * stride_bn)
        -- line 343: the mask `offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K` (`int32`, signed),
        -- broadcast to the pointers' shape; `other=0.0` is the `float32` `+0.0` (a tensor of its
        -- own type), broadcast and converted to the pointer's dtype `float16` (`tl.convert`)
        let a ← tl.load a_ptrs
          (some ((offs_k_for_mask.row BLOCK_SIZE_M).ltS (K - ki * (IntV.ofNat BLOCK_SIZE_K : I32))))
          (some (← tl.convert .f32 .f16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] (FloatV.litF32 (FloatV.ofPat 0)))))
        -- line 344: the mask `offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K`, `other=0.0` as above
        let b ← tl.load b_ptrs
          (some ((offs_k_for_mask.col BLOCK_SIZE_N).ltS (K - ki * (IntV.ofNat BLOCK_SIZE_K : I32))))
          (some (← tl.convert .f32 .f16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] (FloatV.litF32 (FloatV.ofPat 0)))))
        -- line 345: `tl.dot(a, b, accumulator)` of `float16` operands into the `float32`
        -- accumulator; no `input_precision` (the default `tf32`) and no `max_num_imprecise_acc` (`0`)
        tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
          a b accumulator
      -- line 347: `tile_id_c += NUM_SMS`, the constexpr built in `int32`; wrapping `+`
      let tile_id_c := tile_id_c + (IntV.ofNat NUM_SMS : I32)
      -- line 348: `_compute_pid` inlined again, on `tile_id_c`
      let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m GROUP_SIZE_M
      -- lines 349–350
      let offs_cm : Tensor [BLOCK_SIZE_M] I32 := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
      let offs_cn : Tensor [BLOCK_SIZE_N] I32 := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
      -- line 351: `(c_ptr + stride_cm * offs_cm[:, None]) + stride_cn * offs_cn[None, :]`: two
      -- pointer additions, each of an `int32` product, broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_N]`
      let c_ptrs := c_ptr + stride_cm * offs_cm.col BLOCK_SIZE_N + stride_cn * offs_cn.row BLOCK_SIZE_M
      -- line 352: two signed `int32` comparisons with the scalars `M`, `N`, broadcast, then `&`
      let c_mask := ((offs_cm.col BLOCK_SIZE_N).ltS M).and ((offs_cn.row BLOCK_SIZE_M).ltS N)
      -- lines 353, 355–356: the compile-time condition is false; `float32` to `float16`, one
      -- rounding to nearest even (`tl.cvt`)
      let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F16 ← tl.cvt accumulator
      -- line 357: `c` is already `float16`, the pointer's dtype
      tl.store c_ptrs c (some c_mask)
      return tile_id_c

/-! ## The conditions -/

/-- What the host chooses: one launch of `matmul_kernel_persistent` through
`matmul_persistent(a, b)`. -/
structure Inst where
  /-- `M`: the rows of `a` and of `c` (`a.shape[0]`). -/
  M : Nat
  /-- `N`: the columns of `b` and of `c` (`b.shape[1]`). -/
  N : Nat
  /-- `K`: the columns of `a`, the rows of `b` (`a.shape[1]`). -/
  K : Nat
  /-- `NUM_SMS`: the constexpr the host passes, the device's multiprocessor count
  (`torch.cuda.get_device_properties("cuda").multi_processor_count`, line 364; `132` on the H200);
  also the bound of the grid, `min(NUM_SMS, tiles)`. -/
  NUM_SMS : Nat
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

/-- The number of tiles, `cdiv(M, 128) · cdiv(N, 256)` (line 317). -/
def tiles (d : Inst) : Nat := (d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256)

/-- The elements of `a`'s storage that make up the matrix `a` (`M × K`). -/
def aIdx (d : Inst) : Set Nat :=
  {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.stride_am + k * d.stride_ak}

/-- The elements of `b`'s storage that make up the matrix `b` (`K × N`). -/
def bIdx (d : Inst) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.N ∧ e = k * d.stride_bk + j * d.stride_bn}

/-- The elements of `c`'s storage that make up the matrix `c` (`M × N`). -/
def cIdx (d : Inst) : Set Nat :=
  {e | ∃ i j, i < d.M ∧ j < d.N ∧ e = i * d.stride_cm + j * d.stride_cn}

/-- The instances covered. Shapes, strides, layouts and `NUM_SMS` are variables; `M` and `N` are
whole tiles (the hints of lines 334–335 are false on a partial tile); the bounds keep the kernel's
`int32` arithmetic from wrapping, the tile loop's bounds valid, the tolerance meaningful
(`K ≤ 2^13`), and the elements of `c` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M` is a multiple of `BLOCK_SIZE_M = 128`: on a partial tile `offs_am` (line 332) is not
  contiguous, and the hint pair of line 334 is false. -/
  M_tile : d.M % 128 = 0
  /-- `N` is a multiple of `BLOCK_SIZE_N = 256` (line 335, likewise). -/
  N_tile : d.N % 256 = 0
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and `tl.cdiv(M, 128)` computes `M + 127` in `int32`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 256 ≤ 2^31`: likewise for `tl.cdiv(N, 256)`. -/
  N_le : d.N + 256 ≤ 2 ^ 31
  /-- `NUM_SMS` is positive (the step of the tile loop, line 326) and an `int32` constant. -/
  sms : 0 < d.NUM_SMS ∧ d.NUM_SMS < 2 ^ 31
  /-- `tiles + NUM_SMS ≤ 2^31`: the `int32` `num_tiles` (line 317) does not wrap, and the tile
  loop's bounds are valid (`tl.RangeOk`: `num_tiles + NUM_SMS − 1` is an `int32`). -/
  tiles_le : tiles d + d.NUM_SMS ≤ 2 ^ 31
  /-- `K ≤ 2^13`: the tolerance of `Post` grows with `K`; up to `2^13` it is below `0.21%` of the
  products' lifted magnitudes (`tol_rel_lt`). The setup's largest `K` is `1024`. -/
  K_le : d.K ≤ 2 ^ 13
  /-- `N` is a model dimension (the output features): a positive multiple of `128`, at most `2^17`
  (README, Exclusions). -/
  N_dim : 0 < d.N ∧ d.N % 128 = 0 ∧ d.N ≤ 2 ^ 17
  /-- `K` is a model dimension (the inner features): a positive multiple of `128`, at most `2^17`
  (README, Exclusions). -/
  K_dim : 0 < d.K ∧ d.K % 128 = 0 ∧ d.K ≤ 2 ^ 17
  /-- `M` counts tokens: at most `2^22` (README, Exclusions). -/
  M_ctx : d.M ≤ 2 ^ 22
  /-- Each stride is an `int32`. -/
  am : d.stride_am < 2 ^ 31
  /-- As `am`. -/
  ak : d.stride_ak < 2 ^ 31
  /-- As `am`. -/
  bk : d.stride_bk < 2 ^ 31
  /-- As `am`. -/
  bn : d.stride_bn < 2 ^ 31
  /-- As `am`. -/
  cm : d.stride_cm < 2 ^ 31
  /-- As `am`. -/
  cn : d.stride_cn < 2 ^ 31
  /-- The `int32` offsets of `a_ptrs` (line 340), `i * stride_am + k * stride_ak`, do not wrap on
  the lanes the mask of line 343 enables (`i < M`, `k < K`); the disabled lanes' offsets may wrap,
  and are not accessed. -/
  a_fit : (d.M - 1) * d.stride_am + (d.K - 1) * d.stride_ak < 2 ^ 31
  /-- The `int32` offsets of `b_ptrs` (line 341), `k * stride_bk + j * stride_bn`, do not wrap on
  the lanes the mask of line 344 enables (`k < K`, `j < N`). -/
  b_fit : (d.K - 1) * d.stride_bk + (d.N - 1) * d.stride_bn < 2 ^ 31
  /-- The `int32` products `stride_cm * offs_cm` of line 351 do not wrap on the rows below `M`. -/
  cm_fit : (d.M - 1) * d.stride_cm < 2 ^ 31
  /-- The `int32` products `stride_cn * offs_cn` of line 351 do not wrap on the columns below `N`. -/
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

/-- The model on an instance: the buffers' first elements, `M`, `N`, `K` and the strides as
`int32`, the config's block sizes (line 378): `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`,
`BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`, and the instance's `NUM_SMS`. An integer argument equal to
`1` is the compile-time constant `1` (`M` and `N` never are: they are multiples of the blocks);
every use of `K` and of a stride then makes it an `int32`, so the model is the same: an operand of
`+`, `-` or `*` with an `int32` tensor builds it in `int32` (`K - ki * BLOCK_SIZE_K`, lines 343–344;
`offs_k * stride_ak`, `stride_cn * offs_cn`, …, lines 340–341 and 351); `tl.cdiv(K, BLOCK_SIZE_K)`
of two constants (line 316) is Python's `1`, which the name `k_tiles` it is assigned to holds as the
`int32` `1`, as the runtime `int32` `1` gives. A `0` is a runtime `int32`. -/
def model (d : Inst) : Kernel :=
  matmul_kernel_persistent (d.a.ptr 0) (d.b.ptr 0) (d.c.ptr 0)
    (IntV.ofNat d.M) (IntV.ofNat d.N) (IntV.ofNat d.K)
    (IntV.ofNat d.stride_am) (IntV.ofNat d.stride_ak) (IntV.ofNat d.stride_bk) (IntV.ofNat d.stride_bn)
    (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn) 128 256 64 8 d.NUM_SMS

/-- The host's launch (lines 371–379): a 1-D grid of
`min(NUM_SMS, triton.cdiv(M, 128) * triton.cdiv(N, 256))` programs; the source's block shapes;
`enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False`
and `debug = false`. `num_warps=8` and `num_stages=3` are not recorded (performance-only, and
harmless to a well-behaved model). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 (min d.NUM_SMS (tiles d))
    [[128], [256], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 256], [64, 256], [128, 256]]
    (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present). Any `float16` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.c.Valid init

/-! ## The specification -/

/-- The number of `tl.dot` steps of the inner loop (line 338): `cdiv(K, 64)`. -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- The per-step relative constant: `relDot16 64 = 66·2^-22`, the library's constant for one fp16
`tl.dot` of `K = 64` (`Rules.Num.DotF16`). -/
def rho : ℚ := Rules.Num.relDot16 64

/-- The per-step absolute constant: `absDot16 64 = 66·2^-126`. -/
def alpha : ℚ := Rules.Num.absDot16 64

/-- The relative factor `(1 + ρ)^T − 1` on covered instances (`K ≤ 2^13`, `Inst.ok.K_le`): below
`0.21%`. The tolerance `tol` is at most `((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α` with `A` the sum of all
the products' lifted magnitudes (README, Tolerances). -/
theorem tol_rel_lt {K : Nat} (hK : K ≤ 2 ^ 13) :
    (1 + rho) ^ steps K - 1 < 21 / 10000 := by
  have hT : steps K ≤ 128 := by unfold steps; omega
  have h1 : (1 : ℚ) ≤ 1 + rho := by unfold rho Rules.Num.relDot16; norm_num
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : (1 + rho) ^ 128 < 1 + 21 / 10000 := by
    unfold rho Rules.Num.relDot16; norm_num
  linarith

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- Product `e` of the reduction: `va e · vb e` for `e < K`, and `0` from `K` on (the lanes the
masks of lines 343–344 disable, loaded as `+0`). -/
def prodV (va vb : Nat → F16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The lifted magnitude of product `e`: `liftMag (va e) · liftMag (vb e)` for `e < K`, `0` from `K`
on. `Rules.Num.liftMag .f16 q` is `|q|` for zero and normal values and `2^-14` (the smallest normal
magnitude) for a nonzero subnormal one: the magnitude by which the tensor core aligns an operand,
which `Rules.Num.DotF16` counts by. -/
def liftV (va vb : Nat → F16) (K e : Nat) : ℚ :=
  if e < K then Rules.Num.liftMag .f16 (va e).toRat * Rules.Num.liftMag .f16 (vb e).toRat else 0

/-- The exact partial sum before step `t` of a sequence of products `p`: `p e` for `e < 64·t`. -/
def partSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), p e

/-- The lifted magnitude of step `t`: `Σ q e` for `64·t ≤ e < 64·t + 64`. -/
def stepMag (q : Nat → ℚ) (t : Nat) : ℚ := ∑ k : Fin 64, q (64 * t + k)

/-- **The error bound after `t` steps** of the inner loop, from the exact partial sums: `E 0 = 0`
and `E (t + 1) = (1 + ρ)·E t + ρ·(|S t| + L t) + α`, with `S t = partSum p t` the exact partial sum
before step `t` and `L t = stepMag q t` the step's lifted magnitude. It is what `Rules.Num.DotF16`
gives step by step: step `t`'s `tl.dot` of `64` products, with the accumulator `acc` as its own,
leaves a value within `ρ·(|acc| + L t) + α` of `acc + c t` (`c t` the step's exact sum), and with
`acc` within `E t` of `S t`, `|acc| ≤ |S t| + E t`, so the new accumulator is within `E (t + 1)` of
`S (t + 1)`. In closed form `E T = Σ_{t < T} (1 + ρ)^(T − 1 − t)·(ρ·(|S t| + L t) + α)`. -/
def errAt (p q : Nat → ℚ) : Nat → ℚ
  | 0 => 0
  | t + 1 => (1 + rho) * errAt p q t + rho * (|partSum p t| + stepMag q t) + alpha

/-- **The tolerance of `Post`** for row values `va`, column values `vb` and the reduction length
`K`: the error bound after the inner loop's `steps K` steps, `errAt (prodV va vb K) (liftV va vb K)
(steps K)`. It is explicit and computable from the inputs. -/
def tol (va vb : Nat → F16) (K : Nat) : ℚ := errAt (prodV va vb K) (liftV va vb K) (steps K)

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when row `i` of
`a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is
the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` within
`E = tol va vb K` of the exact sum `S = Σ va k · vb k`; and when there is room in `float16`
(`|S| + E ≤ 65504`), `c` is finite and within `E` plus one `float16` rounding
(`(|S| + E)·2^-11 + 2^-25`) of `S`. -/
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
  /-- [rule] `Num.DotF16`: `tl.dot` of `float16` operands into `float32` (`maxImpreciseAcc = 0`,
  any `input_precision`, the model's line 345; `Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at the smallest normal `2^-14`), when that leaves room below the
  `float32` range. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 345) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }


/-! ## The claim -/

/-- The claim to prove: under the assumptions, the launch meets the specification on every
covered instance and input. Not proved here. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotF16, Platform.ref.dotTotal _⟩⟩

/-! ## Coverage: the launches of `SETUP.md`

`hostInst M N K S` is the instance `matmul_persistent(a, b.T)` builds, as the file's `validate` and
`bench` call it, with `NUM_SMS = S`: a contiguous `float16` `a` (`M × K`, strides `(K, 1)`), `b.T`
the transpose of a contiguous `N × K` matrix (`K × N`, strides `(1, K)`), and the fresh `c` it
allocates (`M × N`, strides `(N, 1)`). The three buffers sit at fixed, aligned, far-apart addresses;
the claim covers any addresses `Inst.ok` allows, and these only witness the sizes. -/

/-- The host's instance for the sizes `(M, N, K)` and `NUM_SMS = S`. -/
def hostInst (M N K S : Nat) : Inst where
  M := M
  N := N
  K := K
  NUM_SMS := S
  stride_am := K
  stride_ak := 1
  stride_bk := 1
  stride_bn := K
  stride_cm := N
  stride_cn := 1
  a := ⟨2 ^ 32, M * K⟩
  b := ⟨2 ^ 34, N * K⟩
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
theorem hostInst_ok {M N K S : Nat}
    (h : 0 < M ∧ 0 < N ∧ 0 < K ∧ M % 128 = 0 ∧ N % 256 = 0 ∧ M + 128 ≤ 2 ^ 31 ∧ N + 256 ≤ 2 ^ 31 ∧
      0 < S ∧ S < 2 ^ 31 ∧ (M + 128 - 1) / 128 * ((N + 256 - 1) / 256) + S ≤ 2 ^ 31 ∧ K ≤ 2 ^ 13 ∧
      (M - 1) * K + (K - 1) < 2 ^ 31 ∧ (K - 1) + (N - 1) * K < 2 ^ 31 ∧ (M - 1) * N < 2 ^ 31 ∧
      M * K ≤ 2 ^ 31 ∧ N * K ≤ 2 ^ 31 ∧ M * N ≤ 2 ^ 31 ∧ N ≤ 2 ^ 17 ∧ K % 128 = 0 ∧ M ≤ 2 ^ 22) :
    (hostInst M N K S).ok := by
  obtain ⟨hM, hN, hK, hMt, hNt, h1, h2, hS, hS', ht, h4, h5, h6, h7, hA, hB, hC, hNd, hKd, hMc⟩ := h
  have hal : ∀ i k, i < M → k < K → i * K + k * 1 < M * K := fun i k hi hk => by
    rw [Nat.mul_one]; exact row_lt hi hk
  have hbl : ∀ k j, k < K → j < N → k * 1 + j * K < N * K := fun k j hk hj => by
    rw [Nat.mul_one, Nat.add_comm]; exact row_lt hj hk
  have hcl : ∀ i j, i < M → j < N → i * N + j * 1 < M * N := fun i j hi hj => by
    rw [Nat.mul_one]; exact row_lt hi hj
  have hcS : (hostInst M N K S).c.elems (cIdx (hostInst M N K S)) ⊆ (hostInst M N K S).c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; exact hcl i j hi hj)
  have haS : (hostInst M N K S).a.elems (aIdx (hostInst M N K S)) ⊆ (hostInst M N K S).a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; exact hal i k hi hk)
  have hbS : (hostInst M N K S).b.elems (bIdx (hostInst M N K S)) ⊆ (hostInst M N K S).b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; exact hbl k j hk hj)
  refine ⟨hMt, hNt, h1, h2, ⟨hS, hS'⟩, by simp only [tiles, hostInst]; omega, h4,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega, hMc,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega, by simp only [hostInst]; omega,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega, by simp only [hostInst]; omega,
    by simp only [hostInst, Nat.mul_one]; omega, by simp only [hostInst, Nat.mul_one]; omega,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega,
    fun i j i' j' _ hj _ hj' he => rm_inj hj hj' (by simpa only [hostInst, Nat.mul_one] using he),
    hal, hbl, hcl, ?_, ?_⟩
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega

/-- `K = 128`, `NUM_SMS = 132` (the H200): the test `validate(8192, 8192, 128)` (line 765, with the
module docstring's `--K_range 128 1024 --K_step 128`, line 31; the call of line 715) and the
benchmark's `bench(128)` (lines 668–692, the call of line 682). -/
def setup_k128 : Inst := hostInst 8192 8192 128 132
theorem setup_k128_ok : setup_k128.ok := hostInst_ok (by norm_num)

/-- `K = 256`: the benchmark's `bench(256)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k256 : Inst := hostInst 8192 8192 256 132
theorem setup_k256_ok : setup_k256.ok := hostInst_ok (by norm_num)

/-- `K = 384`: the benchmark's `bench(384)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k384 : Inst := hostInst 8192 8192 384 132
theorem setup_k384_ok : setup_k384.ok := hostInst_ok (by norm_num)

/-- `K = 512`, the default `-K 512` (line 747): the test `validate(8192, 8192, 512)` (line 765) and
the benchmark's `bench(512)` (lines 668–692). -/
def setup_k512 : Inst := hostInst 8192 8192 512 132
theorem setup_k512_ok : setup_k512.ok := hostInst_ok (by norm_num)

/-- `K = 640`: the benchmark's `bench(640)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k640 : Inst := hostInst 8192 8192 640 132
theorem setup_k640_ok : setup_k640.ok := hostInst_ok (by norm_num)

/-- `K = 768`: the benchmark's `bench(768)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k768 : Inst := hostInst 8192 8192 768 132
theorem setup_k768_ok : setup_k768.ok := hostInst_ok (by norm_num)

/-- `K = 896`: the benchmark's `bench(896)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k896 : Inst := hostInst 8192 8192 896 132
theorem setup_k896_ok : setup_k896.ok := hostInst_ok (by norm_num)

/-- `K = 1024`: the benchmark's `bench(1024)` (`--K_range 128 1024 --K_step 128`, lines 668–692). -/
def setup_k1024 : Inst := hostInst 8192 8192 1024 132
theorem setup_k1024_ok : setup_k1024.ok := hostInst_ok (by norm_num)

/-! ## Not vacuous: a covered instance and memory -/

/-- The all-zero memory: every byte present and `0`. -/
def zeroMem : Mem := fun _ => some (0 : TVA.Math.Bytes.Byte)

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_k512, zeroMem, setup_k512_ok, ?_, ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [setup_k512, hostInst], by norm_num [setup_k512, hostInst, DType.bytes],
      by norm_num [setup_k512, hostInst, DType.bytes], fun _ _ => by simp [zeroMem]⟩

end Claims.PersistentMatmul
