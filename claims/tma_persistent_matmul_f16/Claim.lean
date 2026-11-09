import TVA

/-!
# `tma_persistent_matmul_f16`: the prepared claim

Source: `source/09-persistent-matmul.py` (tutorial 09, with the config of `SETUP.md`). The kernel is
`matmul_kernel_tma_persistent` (lines 407–463) with its `@triton.jit` helper `_compute_pid`
(lines 291–298); it reads the module-level names `triton` (its decorator) and `tl` (lines 44–45).
Its host function `matmul_tma_persistent(a, b, False)` (lines 466–503) builds three host
`TensorDescriptor`s (`a` `M × K`, `b` `N × K`, a new `c` `M × N`, all `float16`) with the block
shapes the config's pre-hook sets (`[128, 64]`, `[256, 64]`, `[128, 128]`), and launches
`min(NUM_SMS, tiles)` programs, each looping over the `128 × 256` tiles
`start_pid, start_pid + NUM_SMS, …` of `c = a · bᵀ` and storing each tile as two `128 × 128` halves
(`EPILOGUE_SUBTILE = True`). The launches of `SETUP.md` are the file's test (`M = N = K = 32`, and
`M = N = 8192`) and its benchmark (`M = N = 8192`, `K = 128, 256, …, 1024`).

This file holds the model (`compute_pid`, `matmul_kernel_tma_persistent`, `model`, `launch`), the
conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`), the assumptions
(`Assumptions`) and the claim `Statement`, which it does not prove. It proves `consistent`,
`satisfiable`, and that each launch of `SETUP.md` is covered (`setup_*_ok`).
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-! ## The model -/

/-- The helper `_compute_pid` (source lines 291–298), inlined at each call (lines 433 and 445).
`tile_id`, `num_pid_in_group` and `num_pid_m` are the `int32` scalars the calls pass;
`GROUP_SIZE_M` is the kernel's constexpr, passed directly, so it stays a compile-time value in the
body. The helper's last parameter `NUM_SMS` (the kernel's constexpr) is not used by its body, and
the model has no parameter for it. -/
def compute_pid (tile_id num_pid_in_group num_pid_m : I32) (GROUP_SIZE_M : Nat) : Prog (I32 × I32) := do
  -- line 293: `int32` `//` (`tl.idiv`)
  let group_id := (← tl.idiv (Tensor.scalar tile_id) (Tensor.scalar num_pid_in_group)).item
  -- line 294: `int32` scalar times the constexpr, built in `int32`; wrapping `*`
  let first_pid_m := group_id * (IntV.ofNat GROUP_SIZE_M : I32)
  -- line 295: Python's `min` with a tensor argument is `tl.minimum`, which makes the constexpr an
  -- `int32` tensor of its own type; on `int32` it is the exact `tl.iminimum`
  let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
    (Tensor.scalar (IntV.ofNat GROUP_SIZE_M : I32))).item
  -- line 296: `int32` `%` (`tl.irem`), then the wrapping `+`
  let pid_m := first_pid_m + (← tl.irem (Tensor.scalar tile_id) (Tensor.scalar group_size_m)).item
  -- line 297: `int32` `%`, then `int32` `//`
  let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar tile_id)
    (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
  -- line 298: the pair, which the call sites unpack
  return (pid_m, pid_n)

/-- `matmul_kernel_tma_persistent` (source lines 407–463), with the compile-time values of every
covered launch: `FP8_OUTPUT = False` (the host passes `dtype == torch.float8_e4m3fn` with
`float16` matrices, line 499), so line 419's conditional expression evaluates only its `else` side
and `dtype` holds the compile-time dtype `tl.float16`, which the model writes as `F16` at its uses
(lines 457 and 459); `EPILOGUE_SUBTILE = True` (line 501), so the compile-time `if` of line 453
visits only its first branch (lines 454–460); `WARP_SPECIALIZE = False` (the host's
`warp_specialize`, `False` on the H200: SETUP.md), read only as a scheduling keyword of
`tl.range` (line 432). None of the three is a parameter of the model.

The descriptor parameters receive the host's `TensorDescriptor`s (`TensorDesc .f16`, with the
host's shape, strides, block shape and padding). `M`, `N`, `K` are `int32`: an integer argument in
`[-2^31, 2^31)` other than `1` is a runtime `int32`, and each is below `2^31` on covered instances
(`Inst.ok`); at `1` an argument is the compile-time constant `1`, which gives the same model (see
`model`). The constexprs `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `NUM_SMS`
are Lean parameters.

`hsub`: the reshape of line 454 to `[BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2]` keeps the number of
elements of the `[BLOCK_SIZE_M, BLOCK_SIZE_N]` accumulator; Triton rejects the kernel on the host
otherwise (`tl.reshape`, SEMANTICS.md, "Shape changes"), so the model is defined for the
configurations where it holds (`BLOCK_SIZE_N = 256` on every covered launch). -/
def matmul_kernel_tma_persistent (a_desc b_desc c_desc : TensorDesc .f16) (M N K : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat)
    (hsub : [BLOCK_SIZE_M, 2, BLOCK_SIZE_N / 2].prod = [BLOCK_SIZE_M, BLOCK_SIZE_N].prod) : Kernel where
  body ctx := do
    -- line 419: compile-time (see the docstring); nothing runs
    -- line 420: axis 0 of the 1-D grid, an `int32`
    let start_pid := tl.program_id ctx 0
    -- line 421: `tl.cdiv(M, BLOCK_SIZE_M)` is `(M + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: the
    -- constexpr `BLOCK_SIZE_M - 1` is computed by Python, then built in `int32` (an operand of `+`
    -- with the `int32` `M`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_m := (← tl.idiv (Tensor.scalar (M + (IntV.ofNat (BLOCK_SIZE_M - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_M : I32))).item
    -- line 422: the same with `N`, `BLOCK_SIZE_N`
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 423: the same with `K`, `BLOCK_SIZE_K`
    let k_tiles := (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
    -- line 424: wrapping `int32` `*`
    let num_tiles := num_pid_m * num_pid_n
    -- line 426: the `int32` `start_pid` minus the constexpr, built in `int32`; wrapping `-`
    let tile_id_c := start_pid - (IntV.ofNat NUM_SMS : I32)
    -- line 427: the constexpr times the `int32` scalar, built in `int32`; wrapping `*`
    let num_pid_in_group := (IntV.ofNat GROUP_SIZE_M : I32) * num_pid_n
    -- line 432: `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=False)`:
    -- `int32` bounds, the constexpr step built in `int32`, an `int32` induction variable; `flatten`
    -- and `warp_specialize` are scheduling keywords (no meaning). The loop carries `tile_id_c`, the
    -- only name defined before it that its body assigns (line 444); nothing after the loop uses it
    let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat NUM_SMS : I32) tile_id_c
        fun tile_id tile_id_c => do
      -- line 433: `_compute_pid` inlined, its tuple unpacked
      let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m GROUP_SIZE_M
      -- lines 434–435: `int32` scalar times the constexpr
      let offs_am := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)
      let offs_bn := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)
      -- line 437: `+0.0` in every element
      let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
      -- line 438: `range(k_tiles)`: one argument, the name `k_tiles` evaluated twice (no
      -- operation); `int32` bounds `0`, `k_tiles` and step `1`, an `int32` induction variable; the
      -- loop carries `accumulator` (the only name defined before it that its body assigns)
      let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
          fun ki accumulator => do
        -- line 439: `int32` `ki * BLOCK_SIZE_K`
        let offs_k := ki * (IntV.ofNat BLOCK_SIZE_K : I32)
        -- line 440: `a_desc`'s block `[BLOCK_SIZE_M, BLOCK_SIZE_K]` at the `int32` offsets; lanes
        -- outside the tensor hold the padding (`"zero"`)
        let a ← tl.desc_load (bs := [BLOCK_SIZE_M, BLOCK_SIZE_K]) a_desc [offs_am, offs_k]
        -- line 441: `b_desc`'s block `[BLOCK_SIZE_N, BLOCK_SIZE_K]`
        let b ← tl.desc_load (bs := [BLOCK_SIZE_N, BLOCK_SIZE_K]) b_desc [offs_bn, offs_k]
        -- line 442: `tl.dot(a, b.T, accumulator)`; `b.T` is `b.trans`; `float16` operands into
        -- the `float32` accumulator; no `input_precision` (the default `tf32`) and no
        -- `max_num_imprecise_acc` (`0`)
        tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
          a b.trans accumulator
      -- line 444: `tile_id_c += NUM_SMS`, the constexpr built in `int32`; wrapping `+`
      let tile_id_c := tile_id_c + (IntV.ofNat NUM_SMS : I32)
      -- line 445: `_compute_pid` inlined again, on `tile_id_c`
      let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m GROUP_SIZE_M
      -- lines 446–447
      let offs_am_c := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)
      let offs_bn_c := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)
      -- line 453: the compile-time condition is true (see the docstring)
      -- line 454: `tl.reshape` (row-major, `can_reorder=False`): exact, no step; the shape's
      -- `BLOCK_SIZE_N // 2` is Python's
      let acc := accumulator.reshape [BLOCK_SIZE_M, 2, BLOCK_SIZE_N / 2] hsub
      -- line 455: `tl.permute(acc, (0, 2, 1))`: exact
      let acc : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N / 2, 2] F32 := acc.permute [0, 2, 1]
      -- line 456: `tl.split` of the last dimension (of size 2), the pair unpacked: exact
      let (acc0, acc1) := Tensor.split (s := [BLOCK_SIZE_M, BLOCK_SIZE_N / 2]) acc
      -- line 457: `acc0.to(tl.float16)`, one rounding to nearest even (`tl.cvt`)
      let c0 : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N / 2] F16 ← tl.cvt acc0
      -- line 458: `c_desc`'s block `[BLOCK_SIZE_M, BLOCK_SIZE_N // 2]` at the `int32` offsets;
      -- lanes outside the tensor are not stored
      tl.desc_store c_desc [offs_am_c, offs_bn_c] c0
      -- line 459: as line 457
      let c1 : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N / 2] F16 ← tl.cvt acc1
      -- line 460: the constexpr `BLOCK_SIZE_N // 2` (Python's) built in `int32`; wrapping `+`
      tl.desc_store c_desc [offs_am_c, offs_bn_c + (IntV.ofNat (BLOCK_SIZE_N / 2) : I32)] c1
      return tile_id_c

/-! ## The conditions -/

/-- What the host chooses: one launch of `matmul_kernel_tma_persistent` through
`matmul_tma_persistent(a, b, False)`. Element `(i, k)` of the matrix `a` (`M × K`) is element
`i * lda + k` of the buffer `a`, element `(j, k)` of the matrix `b` (`N × K`) element `j * ldb + k`
of `b`, element `(i, j)` of the matrix `c` (`M × N`) element `i * ldc + j` of `c`: the host builds
each descriptor with `TensorDescriptor.from_tensor` (lines 481–483), whose shape and strides are the
tensor's, and whose last stride is `1` (the class rejects any other). -/
structure Inst where
  /-- `M`: the rows of `a` and of `c` (`a.shape[0]`). -/
  M : Nat
  /-- `N`: the rows of `b`, the columns of `c` (`b.shape[0]`). -/
  N : Nat
  /-- `K`: the columns of `a` and of `b` (`a.shape[1]`). -/
  K : Nat
  /-- `NUM_SMS`: the constexpr the host passes, the device's multiprocessor count
  (`torch.cuda.get_device_properties("cuda").multi_processor_count`, line 477; `132` on the
  H200); also the bound of the grid, `min(NUM_SMS, tiles)`. -/
  NUM_SMS : Nat
  /-- `lda`: the row stride of `a` (`a.stride(0)`), in elements. -/
  lda : Nat
  /-- `ldb`: the row stride of `b` (`b.stride(0)`). -/
  ldb : Nat
  /-- `ldc`: the row stride of `c` (`c.stride(0)`). -/
  ldc : Nat
  /-- The storage of `a`, `float16`, from its first element (the descriptor's base). -/
  a : Buf .f16
  /-- The storage of `b`, `float16`. -/
  b : Buf .f16
  /-- The storage of `c`, `float16`. -/
  c : Buf .f16

/-- The number of tiles, `cdiv(M, 128) · cdiv(N, 256)` (line 424). -/
def tiles (d : Inst) : Nat := (d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256)

/-- The elements of `a`'s storage that make up the matrix `a` (`M × K`). -/
def aIdx (d : Inst) : Set Nat := {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.lda + k}

/-- The elements of `b`'s storage that make up the matrix `b` (`N × K`). -/
def bIdx (d : Inst) : Set Nat := {e | ∃ j k, j < d.N ∧ k < d.K ∧ e = j * d.ldb + k}

/-- The elements of `c`'s storage that make up the matrix `c` (`M × N`). -/
def cIdx (d : Inst) : Set Nat := {e | ∃ i j, i < d.M ∧ j < d.N ∧ e = i * d.ldc + j}

/-- The instances covered. Shapes, row strides, addresses and `NUM_SMS` are variables, partial
tiles included (any `M`, `N`, `K` from `1`); the bounds keep the kernel's `int32` arithmetic from
wrapping, the tile loop's bounds valid, the descriptors within Triton's requirements, the tolerance
meaningful (`K ≤ 2^13`), and the output apart from the inputs. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M`, `N`, `K` are positive: the host's `TensorDescriptor` rejects a zero shape entry, and an
  access through a descriptor needs positive ones (`TensorDesc.Valid`). -/
  pos : 0 < d.M ∧ 0 < d.N ∧ 0 < d.K
  /-- `M + 128 ≤ 2^31`: `M` is an `int32`, and `tl.cdiv(M, 128)` computes `M + 127` in `int32`
  (line 421); the descriptors' shape entries are below `2^31`. -/
  M_le : d.M + 128 ≤ 2 ^ 31
  /-- `N + 256 ≤ 2^31`: likewise for `tl.cdiv(N, 256)` (line 422), and the column offset
  `offs_bn_c + 128` of line 460 stays below `2^31`. -/
  N_le : d.N + 256 ≤ 2 ^ 31
  /-- `NUM_SMS` is positive (the step of the tile loop, line 432) and an `int32` constant. -/
  sms : 0 < d.NUM_SMS ∧ d.NUM_SMS < 2 ^ 31
  /-- `tiles + NUM_SMS ≤ 2^31`: the `int32` `num_tiles` (line 424) does not wrap, and the tile
  loop's bounds are valid (`tl.RangeOk`: `num_tiles + NUM_SMS − 1` is an `int32`). -/
  tiles_le : tiles d + d.NUM_SMS ≤ 2 ^ 31
  /-- `K ≤ 2^13`: the tolerance of `Post` grows with `K`; up to `2^13` its constant `2^-15` covers
  the per-step bound's growth (README, Tolerances). The setup's largest `K` is `1024`. -/
  K_le : d.K ≤ 2 ^ 13
  /-- `N` is a model dimension (the output features): a multiple of `128`, at most `2^17` (README,
  Exclusions). -/
  N_dim : d.N % 128 = 0 ∧ d.N ≤ 2 ^ 17
  /-- `K` is a model dimension (the input features): a multiple of `128`, at most `2^17` (README,
  Exclusions). -/
  K_dim : d.K % 128 = 0 ∧ d.K ≤ 2 ^ 17
  /-- `M`, the rows of `a` (the tokens), is a context length: at most `2^22` (README, Exclusions). -/
  M_ctx : d.M ≤ 2 ^ 22
  /-- Each row stride is a multiple of `8` elements (`16` bytes, `TensorDesc.Valid`; the host's
  `TensorDescriptor` asserts it). -/
  ld8 : d.lda % 8 = 0 ∧ d.ldb % 8 = 0 ∧ d.ldc % 8 = 0
  /-- Each row of `c` lies within `ldc`: `N ≤ ldc`, so the elements of the matrix `c` are distinct
  elements of its storage, and no two lanes or programs store to one element. -/
  N_ldc : d.N ≤ d.ldc
  /-- Each matrix starts at a `16`-byte-aligned address (`TensorDesc.Valid`; the host's
  `TensorDescriptor` asserts it). -/
  base_al : d.a.base % 16 = 0 ∧ d.b.base % 16 = 0 ∧ d.c.base % 16 = 0
  /-- `a`'s storage holds every element of the matrix `a`. -/
  a_len : ∀ i k, i < d.M → k < d.K → i * d.lda + k < d.a.len
  /-- `b`'s storage holds every element of the matrix `b`. -/
  b_len : ∀ j k, j < d.N → k < d.K → j * d.ldb + k < d.b.len
  /-- `c`'s storage holds every element of the matrix `c`. -/
  c_len : ∀ i j, i < d.M → j < d.N → i * d.ldc + j < d.c.len
  /-- The matrix `c` shares no byte with the matrix `a`. -/
  c_a : Disjoint (d.c.elems (cIdx d)) (d.a.elems (aIdx d))
  /-- The matrix `c` shares no byte with the matrix `b`. -/
  c_b : Disjoint (d.c.elems (cIdx d)) (d.b.elems (bIdx d))

/-- The host descriptor of `a` (line 481, block shape set at line 495): base `a`, shape `[M, K]`,
strides `[lda, 1]`, block `[128, 64]`, the default padding `"zero"`. -/
def aDesc (d : Inst) : TensorDesc .f16 := ⟨d.a.ptr 0, [d.M, d.K], [d.lda, 1], [128, 64], .zero⟩

/-- The host descriptor of `b` (line 482): shape `[N, K]`, strides `[ldb, 1]`, block `[256, 64]`. -/
def bDesc (d : Inst) : TensorDesc .f16 := ⟨d.b.ptr 0, [d.N, d.K], [d.ldb, 1], [256, 64], .zero⟩

/-- The host descriptor of `c` (line 483): shape `[M, N]`, strides `[ldc, 1]`, block `[128, 128]`
(`[BLOCK_SIZE_M, BLOCK_SIZE_N // 2]`, the pre-hook's block with `EPILOGUE_SUBTILE = True`). -/
def cDesc (d : Inst) : TensorDesc .f16 := ⟨d.c.ptr 0, [d.M, d.N], [d.ldc, 1], [128, 128], .zero⟩

/-- The model on an instance: the three host descriptors, `M`, `N`, `K` as `int32`, the config's
block sizes (line 501): `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`,
`GROUP_SIZE_M = 8`, and the instance's `NUM_SMS`. An integer argument equal to `1` is the
compile-time constant `1`; the model is the same: `tl.cdiv` of two constants (lines 421–423) is
Python's `1`, which the names `num_pid_m`, `num_pid_n`, `k_tiles` hold as the `int32` `1` wherever
an `int32` operand meets them (`num_pid_m * num_pid_n`, line 424, with the `int32` `num_pid_n` or,
when both are constants, Python's `1` given to `tl.range` as an `int32` bound; `GROUP_SIZE_M *
num_pid_n`, line 427, likewise; `num_pid_m - first_pid_m` in the helper, with the `int32`
`first_pid_m`; `range(k_tiles)`, line 438, whose bounds are then Python integers and whose
induction variable is an `int32`), as the runtime `int32` `1` gives; no value here is near the
`int32` range. -/
def model (d : Inst) : Kernel :=
  matmul_kernel_tma_persistent (aDesc d) (bDesc d) (cDesc d)
    (IntV.ofNat d.M) (IntV.ofNat d.N) (IntV.ofNat d.K) 128 256 64 8 d.NUM_SMS (by norm_num)

/-- The host's launch (lines 485–502): a 1-D grid of
`min(NUM_SMS, triton.cdiv(M, 128) * triton.cdiv(N, 256))` programs; the source's block shapes;
`enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False`
and `debug = false`. `num_warps=8` and `num_stages=4` are not recorded (they do not change the
meaning of a well-behaved model). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 (min d.NUM_SMS (tiles d))
    [[128, 256], [128, 64], [256, 64], [64, 256], [128, 2, 128], [128, 128, 2], [128, 128]]
    (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present). Any `float16` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.c.Valid init

/-! ## The specification -/

/-- The number of `tl.dot` steps of the inner loop (line 438): `cdiv(K, 64)`. -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- The exact dot product of row values `va` (of `a`) and `vb` (of `b`) over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- Product `e` of the reduction: `va e · vb e` for `e < K`, and `0` from `K` on (the lanes past
the tensors' last column, which the descriptors' loads pad with zeros). -/
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

/-- The magnitude the inner loop's accumulation carries: over its `steps K` steps, the exact
partial sum before the step plus the step's lifted magnitude, `Σ_{t < steps K} (|S t| + L t)` with
`S t = partSum p t`, `L t = stepMag q t`. It is at least the sum of all the lifted magnitudes, so at
least `|dotSum|`. -/
def pathMag (va vb : Nat → F16) (K : Nat) : ℚ :=
  ∑ t ∈ Finset.range (steps K), (|partSum (prodV va vb K) t| + stepMag (liftV va vb K) t)

/-- **The tolerance of `Post`** for row values `va`, `vb` and the reduction length `K`:
`2^-15·pathMag + 2^-112`. It is above the step-by-step bound `Rules.Num.DotF16` gives along the
exact partial sums (README, Tolerances), and computable from the inputs. -/
def tol (va vb : Nat → F16) (K : Nat) : ℚ := 2 ^ (-15 : ℤ) * pathMag va vb K + 2 ^ (-112 : ℤ)

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when row `i` of
`a` and row `j` of `b` hold finite values `va k`, `vb k` (`k < K`) in the initial memory, `c` is the
`float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` within
`E = tol va vb K` of the exact sum `S = Σ va k · vb k`; and when there is room in `float16`
(`|S| + E ≤ 65504`), `c` is finite. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.ldc + j)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(i * d.lda + k)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(j * d.ldb + k)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum va vb d.K| ≤ tol va vb d.K ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum va vb d.K| + tol va vb d.K ≤ Math.Float.Format.binary16.maxVal → c.isFinite = true)

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
  any `input_precision`, the model's line 442; `Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at the smallest normal `2^-14`), when that leaves room below the
  `float32` range. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 442) always has a result. -/
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

`hostInst M N K S` is the instance `matmul_tma_persistent(a, b, False)` builds, as the file's
`validate` and `bench` call it, with `NUM_SMS = S`: a contiguous `float16` `a` (`M × K`, row stride
`K`), the contiguous `N × K` `b = b.T.contiguous()` (row stride `K`), and the fresh contiguous `c`
it allocates (`M × N`, row stride `N`). The three buffers sit at fixed, aligned, far-apart
addresses; the claim covers any addresses `Inst.ok` allows, and these only witness the sizes. -/

/-- The host's instance for the sizes `(M, N, K)` and `NUM_SMS = S`. -/
def hostInst (M N K S : Nat) : Inst where
  M := M
  N := N
  K := K
  NUM_SMS := S
  lda := K
  ldb := K
  ldc := N
  a := ⟨2 ^ 32, M * K⟩
  b := ⟨2 ^ 34, N * K⟩
  c := ⟨2 ^ 36, M * N⟩

/-- `t·K + k < M·K` for `t < M`, `k < K`. -/
theorem row_lt {t M K k : Nat} (ht : t < M) (hk : k < K) : t * K + k < M * K := by
  have h : (t + 1) * K ≤ M * K := Nat.mul_le_mul_right K ht
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

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
    (h : 0 < M ∧ 0 < N ∧ 0 < K ∧ M + 128 ≤ 2 ^ 31 ∧ N + 256 ≤ 2 ^ 31 ∧
      0 < S ∧ S < 2 ^ 31 ∧ (M + 128 - 1) / 128 * ((N + 256 - 1) / 256) + S ≤ 2 ^ 31 ∧ K ≤ 2 ^ 13 ∧
      K % 8 = 0 ∧ N % 8 = 0 ∧ M * K ≤ 2 ^ 31 ∧ N * K ≤ 2 ^ 31 ∧ M * N ≤ 2 ^ 31 ∧
      N % 128 = 0 ∧ N ≤ 2 ^ 17 ∧ K % 128 = 0 ∧ M ≤ 2 ^ 22) :
    (hostInst M N K S).ok := by
  obtain ⟨hM, hN, hK, h1, h2, hS, hS', ht, h4, hK8, hN8, hA, hB, hC, hNd, hNd', hKd, hMc⟩ := h
  have hal : ∀ i k, i < M → k < K → i * K + k < M * K := fun i k hi hk => row_lt hi hk
  have hbl : ∀ j k, j < N → k < K → j * K + k < N * K := fun j k hj hk => row_lt hj hk
  have hcl : ∀ i j, i < M → j < N → i * N + j < M * N := fun i j hi hj => row_lt hi hj
  have hcS : (hostInst M N K S).c.elems (cIdx (hostInst M N K S)) ⊆ (hostInst M N K S).c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; exact hcl i j hi hj)
  have haS : (hostInst M N K S).a.elems (aIdx (hostInst M N K S)) ⊆ (hostInst M N K S).a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; exact hal i k hi hk)
  have hbS : (hostInst M N K S).b.elems (bIdx (hostInst M N K S)) ⊆ (hostInst M N K S).b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨j, k, hj, hk, rfl⟩; exact hbl j k hj hk)
  refine ⟨⟨hM, hN, hK⟩, h1, h2, ⟨hS, hS'⟩, by simp only [tiles, hostInst]; omega, h4,
    ⟨hNd, hNd'⟩, show K % 128 = 0 ∧ K ≤ 2 ^ 17 from ⟨hKd, by omega⟩, hMc,
    ⟨hK8, hK8, hN8⟩, le_refl N, by simp only [hostInst]; norm_num, hal, hbl, hcl, ?_, ?_⟩
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega

/-- `K = 128`: the test `validate(8192, 8192, 128)` (line 771, with the module docstring's
`--K_range 128 1024 --K_step 128`, line 35; the call of line 735) and the benchmark's `bench(128)`
(lines 674–698, the call of line 694). -/
def setup_k128 : Inst := hostInst 8192 8192 128 132
theorem setup_k128_ok : setup_k128.ok := hostInst_ok (by norm_num)

/-- `K = 256`: the benchmark's `bench(256)` (`--K_range 128 1024 --K_step 128`, lines 674–698). -/
def setup_k256 : Inst := hostInst 8192 8192 256 132
theorem setup_k256_ok : setup_k256.ok := hostInst_ok (by norm_num)

/-- `K = 384`: the benchmark's `bench(384)`. -/
def setup_k384 : Inst := hostInst 8192 8192 384 132
theorem setup_k384_ok : setup_k384.ok := hostInst_ok (by norm_num)

/-- `K = 512`, the default `-K 512` (line 753): the test `validate(8192, 8192, 512)` (line 771) and
the benchmark's `bench(512)` (lines 674–698). -/
def setup_k512 : Inst := hostInst 8192 8192 512 132
theorem setup_k512_ok : setup_k512.ok := hostInst_ok (by norm_num)

/-- `K = 640`: the benchmark's `bench(640)`. -/
def setup_k640 : Inst := hostInst 8192 8192 640 132
theorem setup_k640_ok : setup_k640.ok := hostInst_ok (by norm_num)

/-- `K = 768`: the benchmark's `bench(768)`. -/
def setup_k768 : Inst := hostInst 8192 8192 768 132
theorem setup_k768_ok : setup_k768.ok := hostInst_ok (by norm_num)

/-- `K = 896`: the benchmark's `bench(896)`. -/
def setup_k896 : Inst := hostInst 8192 8192 896 132
theorem setup_k896_ok : setup_k896.ok := hostInst_ok (by norm_num)

/-- `K = 1024`: the benchmark's `bench(1024)`. -/
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

end Claims.TmaPersistentMatmul
