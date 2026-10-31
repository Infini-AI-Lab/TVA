import TVA

/-!
# `grouped_matmul_tma_f16`: the claim's definitions

Source: `source/08-grouped-gemm.py` (tutorial 08, with the config of `SETUP.md`). The kernel is
`grouped_matmul_tma_kernel` (lines 186–278); it reads the module-level names `triton` (its
decorator) and `tl` (lines 32–33). Its host functions `group_gemm_tma_fn` (lines 281–326) and
`triton_tma_perf_fn` (lines 374–378) launch it with `FP8=False` on groups of `float16` matrices:
the test (lines 329–357: four groups, `M = N = K = 1024, 512, 256, 128`) and the two benchmarks
(lines 386–520: four groups of `N × N × N`, `N = 128 … 1024`, and four groups of `M × 8192 × 8192`,
`M = 128 … 1024`).

This module holds the model (`grouped_matmul_tma_kernel`, `model`, `launch`), the conditions
(`Inst`, `Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`) and the assumptions
(`Assumptions`). The proof is in `GroupedMatmulTma.Sched`, `GroupedMatmulTma.Basic`,
`GroupedMatmulTma.Tile`, `GroupedMatmulTma.Program` and `GroupedMatmulTma`.
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## The model -/

/-- `grouped_matmul_tma_kernel` (source lines 186–278) with `FP8 = False`, the only value the host
passes (`torch.float8_e4m3fn == group_A[0].dtype` is `False` for the `float16` matrices of every
covered launch): line 209's conditional expression on the compile-time `FP8` evaluates only its
`else` side, so `dtype` holds the dtype `tl.float16` (a compile-time value, kept as it is), and
`FP8` is no parameter of the model; `dtype` is used at lines 226–228 (`tl.pointer_type(dtype)`) and
271 (`.to(dtype)`), which the model writes with `float16`.

The pointer arguments are tensors: `group_a_ptrs`, `group_b_ptrs`, `group_c_ptrs` are `int64`
tensors of addresses (`torch.tensor` of Python integers), `group_gemm_sizes` and `g_lds` are `int32`
tensors. `group_size` is an `int32` (an integer argument in `[-2^31, 2^31)` other than `1`; at `1`
it is the compile-time constant `1`, which gives the same model, see `model`). The constexprs
`NUM_SM`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` are Lean parameters. -/
def grouped_matmul_tma_kernel (group_a_ptrs group_b_ptrs group_c_ptrs : Ptr .i64)
    (group_gemm_sizes g_lds : Ptr .i32) (group_size : I32)
    (NUM_SM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    -- line 209: `dtype = tl.float16` (`FP8` is `False`), a compile-time value: see the docstring
    -- line 210: axis 0 of the 1-D grid, an `int32`
    let tile_idx := tl.program_id ctx 0
    -- line 211: the Python `0`, assigned to a name, is an `int32` scalar tensor
    let last_problem_end : I32 := IntV.ofInt 0
    -- line 212: `range(group_size)`: from the `int32` `0` to the `int32` `group_size`, step `1`
    -- (`group_size`, a name, is evaluated twice: nothing to run); the loop carries `tile_idx` (the
    -- `if` of line 220 assigns it, through the `while` of line 251) and `last_problem_end` (line
    -- 278), both defined before it
    let (_tile_idx, _last_problem_end) ← tl.forRange (IntV.ofInt 0 : I32) group_size (IntV.ofInt 1 : I32)
        (tile_idx, last_problem_end) fun g (tile_idx, last_problem_end) => do
      -- line 214: the `int32` product `g * 3` (the Python `3` built in `int32`), added to the
      -- `*int32` pointer, then a scalar load
      let gm ← tl.load1 (group_gemm_sizes + g * (IntV.ofInt 3 : I32))
      -- lines 215–216: `(ptr + g * 3) + 1`: the second addition adds the `int32` constant
      let gn ← tl.load1 (group_gemm_sizes + g * (IntV.ofInt 3 : I32) + (IntV.ofInt 1 : I32))
      let gk ← tl.load1 (group_gemm_sizes + g * (IntV.ofInt 3 : I32) + (IntV.ofInt 2 : I32))
      -- line 217: `tl.cdiv(gm, BLOCK_SIZE_M)` is `(gm + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: the
      -- constexpr `BLOCK_SIZE_M - 1` computed by Python, built in `int32` (an operand of `+` with
      -- the `int32` `gm`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
      let num_m_tiles := (← tl.idiv (Tensor.scalar (gm + (IntV.ofNat (BLOCK_SIZE_M - 1) : I32)))
        (Tensor.scalar (IntV.ofNat BLOCK_SIZE_M : I32))).item
      -- line 218: the same with `gn`, `BLOCK_SIZE_N`
      let num_n_tiles := (← tl.idiv (Tensor.scalar (gn + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
        (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
      -- line 219: the wrapping `int32` product
      let num_tiles := num_m_tiles * num_n_tiles
      -- line 220: a runtime `if`; `and` of two `int1` tensors evaluates both (their `&`); `a >= b`
      -- is the signed `b <= a`; `last_problem_end + num_tiles` is the wrapping `int32` sum. The
      -- names the branch defines (lines 222–248) are new to the `if` and used only in it; the
      -- branch assigns `tile_idx` (through the `while`), which keeps its `int32` type, and the
      -- `if` merges it: the taken branch's value
      let tile_idx ← if (IntV.le last_problem_end tile_idx && IntV.lt tile_idx (last_problem_end + num_tiles)) then do
          -- lines 222–224: scalar loads, as lines 214–216
          let lda ← tl.load1 (g_lds + g * (IntV.ofInt 3 : I32))
          let ldb ← tl.load1 (g_lds + g * (IntV.ofInt 3 : I32) + (IntV.ofInt 1 : I32))
          let ldc ← tl.load1 (g_lds + g * (IntV.ofInt 3 : I32) + (IntV.ofInt 2 : I32))
          -- lines 226–228: the `int64` address loaded from `group_*_ptrs + g` (an `*int64` pointer
          -- plus the `int32` `g`), cast to a `*fp16` pointer (`tl.intToPtr`)
          let a_ptr := tl.intToPtr .f16 (← tl.load1 (group_a_ptrs + g))
          let b_ptr := tl.intToPtr .f16 (← tl.load1 (group_b_ptrs + g))
          let c_ptr := tl.intToPtr .f16 (← tl.load1 (group_c_ptrs + g))
          -- lines 230–235: the shape entries `gm`, `gk` are `int32` already; each stride becomes
          -- `int64`: the `int32` `lda` converted (`tl.convert`), and the Python `1`, an `int32`
          -- tensor of its own type, converted; the block shape; the default padding `"zero"`
          let a_desc ← tl.make_tensor_descriptor a_ptr [gm, gk]
            [(← tl.convert .i32 .i64 (Tensor.scalar lda)).item,
              (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 1 : I32))).item]
            [BLOCK_SIZE_M, BLOCK_SIZE_K]
          -- lines 237–242: as lines 230–235, for `b` (`gn × gk`, row stride `ldb`)
          let b_desc ← tl.make_tensor_descriptor b_ptr [gn, gk]
            [(← tl.convert .i32 .i64 (Tensor.scalar ldb)).item,
              (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 1 : I32))).item]
            [BLOCK_SIZE_N, BLOCK_SIZE_K]
          -- lines 243–248: as lines 230–235, for `c` (`gm × gn`, row stride `ldc`)
          let c_desc ← tl.make_tensor_descriptor c_ptr [gm, gn]
            [(← tl.convert .i32 .i64 (Tensor.scalar ldc)).item,
              (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 1 : I32))).item]
            [BLOCK_SIZE_M, BLOCK_SIZE_N]
          -- line 251: a `while` loop with the source's condition (as line 220's); it carries
          -- `tile_idx` (line 275), defined before it; the names its body defines are new to it and
          -- not used after it
          tl.whileLoop tile_idx
            (fun tile_idx => pure (IntV.le last_problem_end tile_idx && IntV.lt tile_idx (last_problem_end + num_tiles)))
            fun tile_idx => do
              -- line 252: `k` holds the `int32` tensor `gk`
              let k := gk
              -- line 254: the wrapping `int32` difference
              let tile_idx_in_gemm := tile_idx - last_problem_end
              -- lines 255–256: `int32` `//` and `%` (`tl.idiv`, `tl.irem`: truncating, undefined on
              -- a zero divisor)
              let tile_m_idx := (← tl.idiv (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar num_n_tiles)).item
              let tile_n_idx := (← tl.irem (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar num_n_tiles)).item
              -- lines 259–260: the `int32` scalar times the constexpr, built in `int32`
              let offs_am := tile_m_idx * (IntV.ofNat BLOCK_SIZE_M : I32)
              let offs_bn := tile_n_idx * (IntV.ofNat BLOCK_SIZE_N : I32)
              -- line 262: `+0.0` in every element
              let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
              -- line 263: `range(0, tl.cdiv(k, BLOCK_SIZE_K))`: the bound `(k + (BLOCK_SIZE_K - 1)) //
              -- BLOCK_SIZE_K` evaluated once, before the loop; `int32` bounds and induction variable;
              -- the loop carries `accumulator` (defined before it)
              let accumulator ← tl.forRange (IntV.ofInt 0 : I32)
                  (← tl.idiv (Tensor.scalar (k + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
                    (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
                  (IntV.ofInt 1 : I32) accumulator fun kk accumulator => do
                -- line 264: the descriptor's block `[BLOCK_SIZE_M, BLOCK_SIZE_K]` at the offsets
                -- `offs_am` and the `int32` `kk * BLOCK_SIZE_K`; lanes outside `gm × gk` hold `+0.0`
                let a : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F16 ←
                  tl.desc_load a_desc [offs_am, kk * (IntV.ofNat BLOCK_SIZE_K : I32)]
                -- line 265: the block `[BLOCK_SIZE_N, BLOCK_SIZE_K]` of `b`
                let b : Tensor [BLOCK_SIZE_N, BLOCK_SIZE_K] F16 ←
                  tl.desc_load b_desc [offs_bn, kk * (IntV.ofNat BLOCK_SIZE_K : I32)]
                -- line 266: `accumulator + tl.dot(a, b.T)`: a dot with the default (zero)
                -- accumulator whose product has no other use, added to `accumulator`, is
                -- `tl.dot_add`; `b.T` is `b.trans`; `float16` operands into `float32`, no
                -- `input_precision` (the default `tf32`) and no `max_num_imprecise_acc` (`0`)
                let accumulator ←
                  tl.dot_add { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } a b.trans accumulator
                return accumulator
              -- lines 268–269: as lines 259–260
              let offs_cm := tile_m_idx * (IntV.ofNat BLOCK_SIZE_M : I32)
              let offs_cn := tile_n_idx * (IntV.ofNat BLOCK_SIZE_N : I32)
              -- line 271: `.to(tl.float16)`, one rounding to nearest even (`tl.cvt`)
              let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F16 ← tl.cvt accumulator
              -- line 272: `c` is already `float16`, the descriptor's dtype; lanes outside `gm × gn`
              -- are dropped
              tl.desc_store c_desc [offs_cm, offs_cn] c
              -- line 275: the constexpr `NUM_SM` built in `int32`; the wrapping `int32` sum
              let tile_idx := tile_idx + (IntV.ofNat NUM_SM : I32)
              return tile_idx
        else pure tile_idx
      -- line 278: the wrapping `int32` sum
      let last_problem_end := last_problem_end + num_tiles
      return (tile_idx, last_problem_end)
    -- the end of the kernel (line 278 is the loop's last line)
    return ()

/-! ## The conditions -/

/-- What the host chooses: one launch of `grouped_matmul_tma_kernel` through `group_gemm_tma_fn` or
`triton_tma_perf_fn`. Group `g` (`g < group_size`) is the product of an `M g × K g` matrix `A g`
and the transpose of an `N g × K g` matrix `B g` (the host passes `B.T.contiguous()`), into an
`M g × N g` matrix `C g`; element `(i, k)` of `A g` is element `i * lda g + k` of the buffer `A g`,
element `(j, k)` of `B g` element `j * ldb g + k` of `B g`, element `(i, j)` of `C g` element
`i * ldc g + j` of `C g` (row-major, row strides `lda g`, `ldb g`, `ldc g`, as the descriptors'
strides `[ld, 1]` say). -/
structure Inst where
  /-- `group_size`: the number of groups (`len(group_A)`). -/
  group_size : Nat
  /-- `NUM_SM`: the number of programs (`num_sms()`, the device's multiprocessor count). -/
  NUM_SM : Nat
  /-- `M g`: the rows of `A g` and of `C g` (`group_gemm_sizes[3 g]`). -/
  M : Nat → Nat
  /-- `N g`: the rows of `B g`, the columns of `C g` (`group_gemm_sizes[3 g + 1]`). -/
  N : Nat → Nat
  /-- `K g`: the columns of `A g` and of `B g` (`group_gemm_sizes[3 g + 2]`). -/
  K : Nat → Nat
  /-- `lda g`: the row stride of `A g`, in elements (`g_lds[3 g]`, `A.stride(0)`). -/
  lda : Nat → Nat
  /-- `ldb g`: the row stride of `B g` (`g_lds[3 g + 1]`). -/
  ldb : Nat → Nat
  /-- `ldc g`: the row stride of `C g` (`g_lds[3 g + 2]`). -/
  ldc : Nat → Nat
  /-- `A g`: the storage of the `float16` matrix `A g`, whose address is `group_a_ptrs[g]`. -/
  A : Nat → Buf .f16
  /-- `B g`: the storage of `B g` (`group_b_ptrs[g]`). -/
  B : Nat → Buf .f16
  /-- `C g`: the storage of `C g` (`group_c_ptrs[g]`). -/
  C : Nat → Buf .f16
  /-- `group_a_ptrs`: an `int64` tensor, the addresses of the `A g`. -/
  group_a_ptrs : Buf .i64
  /-- `group_b_ptrs`: an `int64` tensor, the addresses of the `B g`. -/
  group_b_ptrs : Buf .i64
  /-- `group_c_ptrs`: an `int64` tensor, the addresses of the `C g`. -/
  group_c_ptrs : Buf .i64
  /-- `group_gemm_sizes`: an `int32` tensor, `M g, N g, K g` for each group. -/
  group_gemm_sizes : Buf .i32
  /-- `g_lds`: an `int32` tensor, `lda g, ldb g, ldc g` for each group. -/
  g_lds : Buf .i32

/-- The tiles of group `g`: `cdiv(M g, 128) · cdiv(N g, 128)` (lines 217–219). -/
def tiles (d : Inst) (g : Nat) : Nat := (d.M g + 127) / 128 * ((d.N g + 127) / 128)

/-- The global index of group `g`'s first tile: the tiles of the groups before it (the value of
`last_problem_end` before trip `g` of line 212). -/
def start (d : Inst) (g : Nat) : Nat := ∑ h ∈ Finset.range g, tiles d h

/-- The elements of `A g`'s storage that make up the matrix `A g` (`M g × K g`). -/
def aIdx (d : Inst) (g : Nat) : Set Nat :=
  {e | ∃ i k, i < d.M g ∧ k < d.K g ∧ e = i * d.lda g + k}

/-- The elements of `B g`'s storage that make up the matrix `B g` (`N g × K g`). -/
def bIdx (d : Inst) (g : Nat) : Set Nat :=
  {e | ∃ j k, j < d.N g ∧ k < d.K g ∧ e = j * d.ldb g + k}

/-- The elements of `C g`'s storage that make up the matrix `C g` (`M g × N g`). -/
def cIdx (d : Inst) (g : Nat) : Set Nat :=
  {e | ∃ i j, i < d.M g ∧ j < d.N g ∧ e = i * d.ldc g + j}

/-- The bytes of the matrix `C g`. -/
def cEl (d : Inst) (g : Nat) : Set Int := (d.C g).elems (cIdx d g)

/-- The bytes of the five tables the kernel reads: the first `group_size` elements of each
pointer table, the first `3 · group_size` of `group_gemm_sizes` and `g_lds`. -/
def tabEl (d : Inst) : Set Int :=
  d.group_a_ptrs.elems {e | e < d.group_size} ∪ d.group_b_ptrs.elems {e | e < d.group_size} ∪
    d.group_c_ptrs.elems {e | e < d.group_size} ∪
    d.group_gemm_sizes.elems {e | e < 3 * d.group_size} ∪ d.g_lds.elems {e | e < 3 * d.group_size}

/-- The instances covered. Group counts, shapes, strides and the number of programs are
variables; the bounds keep the kernel's `int32` arithmetic from wrapping, the descriptors within
Triton's requirements, the tolerance meaningful (`K g ≤ 2^13`), and the outputs apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `3 · group_size + 3 ≤ 2^31`: `group_size` is an `int32`, and the `int32` offsets `g * 3 + 2`
  of lines 214–224 do not wrap. -/
  G_le : 3 * d.group_size + 3 ≤ 2 ^ 31
  /-- At least one program (`NUM_SM`, the grid of lines 322 and 375). -/
  SM_pos : 0 < d.NUM_SM
  /-- The tiles of all groups and `NUM_SM` stay below `2^31`: `last_problem_end + num_tiles` (lines
  220, 251, 278), `num_m_tiles * num_n_tiles` (line 219) and `tile_idx + NUM_SM` (line 275), all
  `int32`, do not wrap. -/
  tiles_lt : start d d.group_size + d.NUM_SM < 2 ^ 31
  /-- `M g + 128 ≤ 2^31`: `M g` is an `int32`, and `tl.cdiv(gm, 128)` computes `gm + 127` in
  `int32` (line 217); the descriptor's shape entries are `int32`. -/
  M_le : ∀ g, g < d.group_size → d.M g + 128 ≤ 2 ^ 31
  /-- As `M_le`, for `N g` (line 218). -/
  N_le : ∀ g, g < d.group_size → d.N g + 128 ≤ 2 ^ 31
  /-- `K g ≤ 2^13`: the tolerance of `Post` grows with `K`; up to `2^13` it is below `0.21%` of
  `Σ|a·b|` (`tol_rel_lt`), so `Post` still rejects a result rounded to `bfloat16` (whose largest
  relative error is `2^-8/(1 + 2^-8)`, about `0.39%`). The host's largest `K` is `8192 = 2^13`. -/
  K_le : ∀ g, g < d.group_size → d.K g ≤ 2 ^ 13
  /-- `lda g` is an `int32` and a multiple of `8`: the descriptor's row stride is a multiple of
  `16` bytes (`TensorDesc.Valid`). -/
  lda_ok : ∀ g, g < d.group_size → d.lda g < 2 ^ 31 ∧ d.lda g % 8 = 0
  /-- As `lda_ok`, for `ldb g`. -/
  ldb_ok : ∀ g, g < d.group_size → d.ldb g < 2 ^ 31 ∧ d.ldb g % 8 = 0
  /-- As `lda_ok`, for `ldc g`, and each row of `C g` lies within `ldc g`: `N g ≤ ldc g`, so the
  rows do not overlap and no two lanes or programs store to one element. -/
  ldc_ok : ∀ g, g < d.group_size → d.ldc g < 2 ^ 31 ∧ d.ldc g % 8 = 0 ∧ d.N g ≤ d.ldc g
  /-- Each matrix starts at a `16`-byte-aligned address (`TensorDesc.Valid`). -/
  base_al : ∀ g, g < d.group_size → (d.A g).base % 16 = 0 ∧ (d.B g).base % 16 = 0 ∧ (d.C g).base % 16 = 0
  /-- `A g`'s storage holds every element of the matrix `A g`. -/
  A_len : ∀ g, g < d.group_size → ∀ i k, i < d.M g → k < d.K g → i * d.lda g + k < (d.A g).len
  /-- `B g`'s storage holds every element of the matrix `B g`. -/
  B_len : ∀ g, g < d.group_size → ∀ j k, j < d.N g → k < d.K g → j * d.ldb g + k < (d.B g).len
  /-- `C g`'s storage holds every element of the matrix `C g`. -/
  C_len : ∀ g, g < d.group_size → ∀ i j, i < d.M g → j < d.N g → i * d.ldc g + j < (d.C g).len
  /-- Each pointer table has `group_size` elements, each size table `3 · group_size`. -/
  tab_len : d.group_size ≤ d.group_a_ptrs.len ∧ d.group_size ≤ d.group_b_ptrs.len ∧
    d.group_size ≤ d.group_c_ptrs.len ∧ 3 * d.group_size ≤ d.group_gemm_sizes.len ∧
    3 * d.group_size ≤ d.g_lds.len
  /-- The outputs of two groups share no byte. -/
  c_c : ∀ g h, g < d.group_size → h < d.group_size → g ≠ h → Disjoint (cEl d g) (cEl d h)
  /-- No output shares a byte with an input matrix. -/
  c_ab : ∀ g h, g < d.group_size → h < d.group_size →
    Disjoint (cEl d g) ((d.A h).elems (aIdx d h)) ∧ Disjoint (cEl d g) ((d.B h).elems (bIdx d h))
  /-- No output shares a byte with a table. -/
  c_tab : ∀ g, g < d.group_size → Disjoint (cEl d g) (tabEl d)
  /-- No input matrix shares a byte with a table: the kernel reads the matrices through
  descriptors and the tables through pointers, and the library keeps pointer and descriptor
  accesses to the same bytes out of scope (`docs/exclusions.md`). -/
  ab_tab : ∀ h, h < d.group_size →
    Disjoint ((d.A h).elems (aIdx d h)) (tabEl d) ∧ Disjoint ((d.B h).elems (bIdx d h)) (tabEl d)

/-- The model on an instance: the tables' first elements, `group_size` as an `int32`, and the
launch's constexprs (lines 324–325 and 376–378): `NUM_SM`, `BLOCK_SIZE_M = 128`,
`BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64`. With `group_size = 1` Triton passes the compile-time
constant `1`; its only use is `range(group_size)` (line 212), whose bounds `0` and `1` and step are
then Python integers, so the induction variable is an `int32` and the loop runs one trip, as the
model's `tl.forRange` from the `int32` `0` to the `int32` `1` gives: the model is the same. -/
def model (d : Inst) : Kernel :=
  grouped_matmul_tma_kernel (d.group_a_ptrs.ptr 0) (d.group_b_ptrs.ptr 0) (d.group_c_ptrs.ptr 0)
    (d.group_gemm_sizes.ptr 0) (d.g_lds.ptr 0) (IntV.ofNat d.group_size) d.NUM_SM 128 128 64

/-- The host's launch (lines 322–325, 375–378): a 1-D grid of `NUM_SM` programs
(`grid = lambda META: (META['NUM_SM'], )`); the source's block shapes, `[128, 64]` (the blocks of
`a` and `b`), `[64, 128]` (`b.T`) and `[128, 128]` (the accumulator, `c` and its block);
`enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False`
and `debug = false`. `num_warps=4` and `num_stages=3` are not recorded (they do not change the
meaning). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.NUM_SM [[128, 64], [64, 128], [128, 128]] (fusion := true)

/-- The inputs covered: every buffer is valid (aligned, inside `[0, 2^64)`, every byte present);
the tables hold the instance's values (`group_gemm_sizes[3 g + 0, 1, 2] = M g, N g, K g`,
`g_lds[3 g + 0, 1, 2] = lda g, ldb g, ldc g`, and `group_*_ptrs[g]` the address of `A g`, `B g`,
`C g`) for each `g < group_size`. Any `float16` values in the matrices, NaN and infinities
included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.group_a_ptrs.Valid init ∧ d.group_b_ptrs.Valid init ∧ d.group_c_ptrs.Valid init ∧
    d.group_gemm_sizes.Valid init ∧ d.g_lds.Valid init ∧
    (∀ g, g < d.group_size → (d.A g).Valid init ∧ (d.B g).Valid init ∧ (d.C g).Valid init) ∧
    (∀ g, g < d.group_size →
      init.read .i32 (d.group_gemm_sizes.ptr ↑(3 * g)).addr = some (IntV.ofNat (d.M g)) ∧
      init.read .i32 (d.group_gemm_sizes.ptr ↑(3 * g + 1)).addr = some (IntV.ofNat (d.N g)) ∧
      init.read .i32 (d.group_gemm_sizes.ptr ↑(3 * g + 2)).addr = some (IntV.ofNat (d.K g)) ∧
      init.read .i32 (d.g_lds.ptr ↑(3 * g)).addr = some (IntV.ofNat (d.lda g)) ∧
      init.read .i32 (d.g_lds.ptr ↑(3 * g + 1)).addr = some (IntV.ofNat (d.ldb g)) ∧
      init.read .i32 (d.g_lds.ptr ↑(3 * g + 2)).addr = some (IntV.ofNat (d.ldc g)) ∧
      init.read .i64 (d.group_a_ptrs.ptr ↑g).addr = some (IntV.ofInt (d.A g).base) ∧
      init.read .i64 (d.group_b_ptrs.ptr ↑g).addr = some (IntV.ofInt (d.B g).base) ∧
      init.read .i64 (d.group_c_ptrs.ptr ↑g).addr = some (IntV.ofInt (d.C g).base))

/-! ## The specification -/

/-- The number of `tl.dot` steps of the loop of line 263: `cdiv(K, 64)`. -/
def steps (K : Nat) : Nat := (K + 63) / 64

/-- The per-step relative error of `acc += tl.dot(a, b.T)` (line 266), for one fp16 `tl.dot` of
`K = 64` products: `DotLoop.stepRel relDot16 64 = relDot16 64 + 3·2^-24·(1 + relDot16 64)`, the
library's constant of `Rules.Num.DotF16` plus one `float32` addition (the combine pass may leave
the addition separate, `tl.dot_add`); about `1.59·10^-5`. -/
def rho : ℚ := DotLoop.stepRel Rules.Num.relDot16 64

/-- The per-step absolute error: `DotLoop.stepAbs absDot16 64 =
absDot16 64·(1 + 3·2^-24) + 4·2^-150`, below `10^-36`. -/
def alpha : ℚ := DotLoop.stepAbs Rules.Num.absDot16 64

/-- The tolerance of the `float32` accumulation, for a reduction length `K` and the sum `A` of the
products' magnitudes: `DotLoop.bound rho alpha T A = ((1 + rho)^T − 1)·A + T·(1 + rho)^T·alpha`,
with `T = steps K` steps of `64` products each. It is about `2.5·10^-4·A` at `K = 1024` (the
test's largest) and `2.04·10^-3·A` at `K = 2^13` (the benchmark's largest and the largest covered;
below `0.21%`, `tol_rel_lt`), plus below `10^-34`. -/
def tol (K : Nat) (A : ℚ) : ℚ := DotLoop.bound rho alpha (steps K) A

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The sum of the magnitudes of the products. -/
def dotAbs (va vb : Nat → F16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, |(va k).toRat * (vb k).toRat|

/-- What the launch leaves in element `(i, j)` of `C g`: some `float16` value `c`; and when row
`i` of `A g` and row `j` of `B g` hold finite values `va k`, `vb k` (`k < K g`) in the initial
memory, `c` is the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32`
value `v` within `tol (K g) A` of the exact sum `S = Σ va k · vb k` (`A = Σ |va k · vb k|`); and
when there is room in `float16` (`|S| + tol ≤ 65504`), `c` is finite and within `tol` plus one
`float16` rounding (`(|S| + tol)·2^-11 + 2^-25`) of `S`. -/
def OutSpec (d : Inst) (init fin : Mem) (g i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 ((d.C g).ptr ↑(i * d.ldc g + j)).addr = some c ∧
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K g → init.read .f16 ((d.A g).ptr ↑(i * d.lda g + k)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K g → init.read .f16 ((d.B g).ptr ↑(j * d.ldb g + k)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum va vb (d.K g)| ≤ tol (d.K g) (dotAbs va vb (d.K g)) ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum va vb (d.K g)| + tol (d.K g) (dotAbs va vb (d.K g)) ≤ Math.Float.Format.binary16.maxVal →
          c.isFinite = true ∧
          |c.toRat - dotSum va vb (d.K g)| ≤ tol (d.K g) (dotAbs va vb (d.K g)) +
            (|dotSum va vb (d.K g)| + tol (d.K g) (dotAbs va vb (d.K g))) * Math.Float.Format.binary16.u +
              Math.Float.Format.binary16.eta)

/-- What the launch leaves: every element of every matrix `C g` as `OutSpec` says; every byte
outside the matrices `C g` unchanged (the inputs, the tables, and the elements of each `C g`'s
storage outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ g i j, g < d.group_size → i < d.M g → j < d.N g → OutSpec d init fin g i j) ∧
    ∀ z, (∀ g, g < d.group_size → z ∉ cEl d g) → fin z = init z

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
  any `input_precision`, the model's line 266): each output element whose row of `a`, column of `b`
  and accumulator are finite is finite and within `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of
  `acc + Σ a·b`, when that leaves room below the `float32` range. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 266) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

end Kernels.GroupedMatmulTma
