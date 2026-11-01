import TVA

/-!
# `block_scaled_matmul_mxfp8`: the claim's definitions

Source: `source/10-block-scaled-matmul.py` (tutorial 10, unchanged; `SETUP.md`). The kernel is
`block_scaled_matmul_kernel` (lines 174–240); it reads the module-level names `triton` (its
decorator) and `tl` (lines 123–124). The host function `block_scaled_matmul` (lines 243–281)
launches it with the config `initialize_block_scaled` (lines 330–449) builds for
`block_scale_type = "mxfp8"` on the H200: `float8e4nv` operands `a` (`M × K`) and `b` (`N × K`),
`e8m0` scales (`uint8`) in groups of `VEC_SIZE = 32` along `K` in the 5-D preshuffled layout, and
a `float16` output `c` (`M × N`). The test (`validate_block_scaled(8192, 8192, 8192, "mxfp8")`,
line 749) and the benchmark (`bench_block_scaled(K, block_scale_type="mxfp8")`, lines 472–502:
`M = N = 8192`, `K` a multiple of 128) make these launches.

This module holds the model (`block_scaled_matmul_kernel`, `model`, `launch`), the conditions
(`Inst`, `Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`, with the tolerance
`tol`) and the assumptions (`Assumptions`). The proof is in `BlockScaledMxfp8.Basic`,
`BlockScaledMxfp8.Program` and `BlockScaledMxfp8`.
-/

namespace Kernels.BlockScaledMxfp8

open TVA

/-! ## The model -/

/-- The `tl.dot_scaled` configuration of line 234: `"e4m3"` by `"e4m3"`, `e8m0` scales on both
sides (`uint8` tensors), `32` elements per scale (the dot's `K = BLOCK_K = 128` over the scales'
last dimension `BLOCK_K // VEC_SIZE = 4`), `K`-packed operands (the defaults `lhs_k_pack=True`,
`rhs_k_pack=True`), and the default `fast_math=False`. -/
def cfgMx : ScaledCfg :=
  { lhs := .e4m3, rhs := .e4m3, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
    lhsKPack := true, rhsKPack := true, fastMath := false }

/-- `block_scaled_matmul_kernel` (source lines 174–240) for the `mxfp8` launches: the constexprs
`ELEM_PER_BYTE_A = ELEM_PER_BYTE_B = 1` and `output_type = 1` (the values `block_scaled_matmul`
passes for `"mxfp8"` with `dtype_dst = torch.float16`, lines 247–248, 269–270, and
`initialize_block_scaled`, lines 336–337) are fixed, so they are no parameters of the model:
lines 197–202 give `output_dtype` the dtype `tl.float16` (a compile-time value, kept as it is),
line 216 gives the annotated `MIXED_PREC` the compile-time `False`, and lines 229–233 take the
`else` branch (line 234). `NUM_STAGES` and `disallow_acc_multi_buffer` are only `tl.range`
scheduling keywords (line 219–220), and `N` is read by no line of the body, so they are no
parameters either. The other constexprs (`M`, `K`, `VEC_SIZE`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K`,
`rep_m`, `rep_n`, `rep_k`) are Lean parameters; `model` passes the host's values.

The descriptor parameters receive the host's `TensorDescriptor`s (`TensorDesc`, with the host's
shape, strides, block shape and padding): `a_desc`, `b_desc` of `float8e4nv`, `a_scale_desc`,
`b_scale_desc` of `uint8`, `c_desc` of `float16`. Each `desc.load` has the descriptor's block
shape, written as the host's (lines 367–368, 399–400; `block_scaled_matmul`'s line 256 for `c`).

`hA` and `hB`: the reshapes of lines 226–227 to `[BLOCK_M, BLOCK_K // VEC_SIZE]` and
`[BLOCK_N, BLOCK_K // VEC_SIZE]` keep the number of elements of the permuted scale blocks
(`[rep_m, 4, 32, rep_k, 4]` and `[rep_n, 4, 32, rep_k, 4]`); Triton rejects the kernel on the
host otherwise (`tl.reshape`, SEMANTICS.md, shape changes), so the model is defined for the
configurations where they hold (on every covered launch: `512 = 512` and `1024 = 1024`). -/
def block_scaled_matmul_kernel (a_desc : TensorDesc .f8e4nv) (a_scale_desc : TensorDesc .u8)
    (b_desc : TensorDesc .f8e4nv) (b_scale_desc : TensorDesc .u8) (c_desc : TensorDesc .f16)
    (M K VEC_SIZE BLOCK_M BLOCK_N BLOCK_K rep_m rep_n rep_k : Nat)
    (hA : [BLOCK_M, BLOCK_K / VEC_SIZE].prod = [rep_m, 4, 32, rep_k, 4].prod)
    (hB : [BLOCK_N, BLOCK_K / VEC_SIZE].prod = [rep_n, 4, 32, rep_k, 4].prod) : Kernel where
  body ctx := do
    -- lines 197–202: a compile-time `if` on `output_type = 1`: `output_dtype` is the dtype
    -- `tl.float16` (see the docstring); nothing runs
    -- line 204: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 205: `tl.cdiv(M, BLOCK_M)` of two constexprs is Python's `(M + (BLOCK_M - 1)) // BLOCK_M`
    -- (a compile-time value); assigned to a name it holds an `int32` scalar tensor (it is below
    -- `2^31`, `Inst.ok.M_ok`)
    let num_pid_m : I32 := IntV.ofNat ((M + (BLOCK_M - 1)) / BLOCK_M)
    -- line 206: the `int32` `%` (`tl.irem`, truncating, undefined on a zero divisor)
    let pid_m := (← tl.irem (Tensor.scalar pid) (Tensor.scalar num_pid_m)).item
    -- line 207: the `int32` `//` (`tl.idiv`)
    let pid_n := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_m)).item
    -- lines 208–209: the `int32` scalar times the constexpr, built in `int32` (wrapping)
    let offs_am := pid_m * (IntV.ofNat BLOCK_M : I32)
    let offs_bn := pid_n * (IntV.ofNat BLOCK_N : I32)
    -- lines 210–211: the Python `0` assigned to a name is an `int32` scalar tensor
    let offs_k_a : I32 := IntV.ofInt 0
    let offs_k_b : I32 := IntV.ofInt 0
    -- lines 212–213: as lines 208–209
    let offs_scale_m := pid_m * (IntV.ofNat rep_m : I32)
    let offs_scale_n := pid_n * (IntV.ofNat rep_n : I32)
    -- line 214: as line 210
    let offs_scale_k : I32 := IntV.ofInt 0
    -- line 216: the annotated `MIXED_PREC` is the compile-time `False` (see the docstring)
    -- line 218: `+0.0` in every element
    let accumulator := tl.zeros [BLOCK_M, BLOCK_N] .f32
    -- lines 219–220: `tl.range(0, tl.cdiv(K, BLOCK_K))`: the bound is Python's
    -- `(K + (BLOCK_K - 1)) // BLOCK_K` (two constexprs), and both bounds become `int32` constants,
    -- so the induction variable `k` is an `int32`, step `1`; the scheduling keywords change no
    -- outcome; the loop carries `accumulator`, `offs_k_a`, `offs_k_b` and `offs_scale_k` (lines
    -- 230–238), all defined before it; `k` is not used, and after the loop only `accumulator` is
    let (accumulator, _offs_k_a, _offs_k_b, _offs_scale_k) ←
      tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + (BLOCK_K - 1)) / BLOCK_K) : I32) (IntV.ofInt 1 : I32)
        (accumulator, offs_k_a, offs_k_b, offs_scale_k) fun _k (accumulator, offs_k_a, offs_k_b, offs_scale_k) => do
      -- line 221: `a_desc`'s block `[BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A]` (`ELEM_PER_BYTE_A = 1`)
      -- at the `int32` offsets; lanes outside the tensor hold the padding (`"zero"`)
      let a ← tl.desc_load (bs := [BLOCK_M, BLOCK_K]) a_desc [offs_am, offs_k_a]
      -- line 222: `b_desc`'s block `[BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B]` (`ELEM_PER_BYTE_B = 1`)
      let b ← tl.desc_load (bs := [BLOCK_N, BLOCK_K]) b_desc [offs_bn, offs_k_b]
      -- line 223: the 5-D `uint8` block `[1, rep_m, rep_k, 2, 256]` (line 399); the Python `0`s of
      -- the offsets are `int32` constants
      let scale_a ← tl.desc_load (bs := [1, rep_m, rep_k, 2, 256]) a_scale_desc
        [(IntV.ofInt 0 : I32), offs_scale_m, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
      -- line 224: as line 223, block `[1, rep_n, rep_k, 2, 256]` (line 400)
      let scale_b ← tl.desc_load (bs := [1, rep_n, rep_k, 2, 256]) b_scale_desc
        [(IntV.ofInt 0 : I32), offs_scale_n, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
      -- line 226: `reshape` (row-major, `can_reorder=False`), `trans(0, 3, 2, 1, 4)` (the permutation
      -- with those `dims`), `reshape` to `[BLOCK_M, BLOCK_K // VEC_SIZE]`: exact, no step
      let scale_a := ((scale_a.reshape [rep_m, rep_k, 32, 4, 4]
        (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
          [BLOCK_M, BLOCK_K / VEC_SIZE] hA
      -- line 227: as line 226, to `[BLOCK_N, BLOCK_K // VEC_SIZE]`
      let scale_b := ((scale_b.reshape [rep_n, rep_k, 32, 4, 4]
        (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
          [BLOCK_N, BLOCK_K / VEC_SIZE] hB
      -- lines 229–233: compile-time conditions (`MIXED_PREC` is `False`; `ELEM_PER_BYTE_A == 2` is
      -- `False` and ends the `and`): the `else` branch
      -- line 234: `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)`: `e8m0`
      -- scales (`uint8`), group `BLOCK_K / (BLOCK_K // VEC_SIZE) = 32`, `K`-packed operands, the
      -- default `fast_math=False`; `b.T` is `b.trans`
      let accumulator ← tl.dot_scaled cfgMx BLOCK_M BLOCK_N BLOCK_K a (some scale_a) b.trans (some scale_b) accumulator
      -- lines 236–237: `BLOCK_K // ELEM_PER_BYTE_*` is Python's `BLOCK_K` (`ELEM_PER_BYTE_* = 1`),
      -- built in `int32`; the wrapping `int32` sum
      let offs_k_a := offs_k_a + (IntV.ofNat BLOCK_K : I32)
      let offs_k_b := offs_k_b + (IntV.ofNat BLOCK_K : I32)
      -- line 238: the constexpr `rep_k` built in `int32`
      let offs_scale_k := offs_scale_k + (IntV.ofNat rep_k : I32)
      return (accumulator, offs_k_a, offs_k_b, offs_scale_k)
    -- line 240: `accumulator.to(output_dtype)` with `output_dtype = tl.float16`, one rounding to
    -- nearest even (`tl.cvt`), evaluated before the store; `c_desc`'s block at `[offs_am, offs_bn]`;
    -- lanes outside the tensor are dropped
    let c : Tensor [BLOCK_M, BLOCK_N] F16 ← tl.cvt accumulator
    tl.desc_store c_desc [offs_am, offs_bn] c

/-! ## The conditions -/

/-- What the host chooses: one launch of `block_scaled_matmul_kernel` through
`block_scaled_matmul(…, torch.float16, M, N, K, …)` on the tensors `initialize_block_scaled(M, N,
K, "mxfp8")` builds (lines 340–404), all contiguous:
* `a`: `M × K` `float8e4nv` (`a_ref.to(torch.float8_e4m3fn)`, line 354), element `(i, k)` at
  `i * K + k`;
* `b`: `N × K` `float8e4nv` (line 361), element `(j, k)` at `j * K + k`;
* `a_scale`: the `uint8` `e8m0` scales of `a` (`MXScaleTensor(…).data`, lines 387–389), shape
  `[M // 128, K // 128, 32, 16]` (line 370), viewed as `[1, M // 128, K // 128, 2, 256]`
  (line 401);
* `b_scale`: those of `b`, `[N // 128, K // 128, 32, 16]` (line 371), viewed as
  `[1, N // 128, K // 128, 2, 256]` (line 402);
* `c`: the `M × N` `float16` output (`torch.empty((M, N), dtype=torch.float16)`, line 244),
  element `(i, j)` at `i * N + j`. -/
structure Inst where
  /-- `M`: the rows of `a` and `c` (a constexpr). -/
  M : Nat
  /-- `N`: the rows of `b`, the columns of `c` (a constexpr). -/
  N : Nat
  /-- `K`: the columns of `a` and `b` (a constexpr). -/
  K : Nat
  /-- The storage of `a`. -/
  a : Buf .f8e4nv
  /-- The storage of `b`. -/
  b : Buf .f8e4nv
  /-- The storage of `a`'s scales. -/
  a_scale : Buf .u8
  /-- The storage of `b`'s scales. -/
  b_scale : Buf .u8
  /-- The storage of the output `c`. -/
  c : Buf .f16

/-- `a_desc` (line 367): `TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])`:
shape `[M, K]`, strides `[K, 1]`, block `[128, 128]`, padding `"zero"`. -/
def aDesc (d : Inst) : TensorDesc .f8e4nv := ⟨d.a.ptr 0, [d.M, d.K], [d.K, 1], [128, 128], .zero⟩

/-- `b_desc` (line 368): shape `[N, K]`, strides `[K, 1]`, block `[256, 128]`, padding `"zero"`. -/
def bDesc (d : Inst) : TensorDesc .f8e4nv := ⟨d.b.ptr 0, [d.N, d.K], [d.K, 1], [256, 128], .zero⟩

/-- `a_scale_desc` (lines 401, 403): the contiguous `[1, M // 128, K // 128, 2, 256]` view, its
strides, block `[1, rep_m, rep_k, 2, 256] = [1, 1, 1, 2, 256]` (line 399), padding `"zero"`. -/
def aScaleDesc (d : Inst) : TensorDesc .u8 :=
  ⟨d.a_scale.ptr 0, [1, d.M / 128, d.K / 128, 2, 256],
    [d.M / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 1, 1, 2, 256], .zero⟩

/-- `b_scale_desc` (lines 402, 404): `[1, N // 128, K // 128, 2, 256]`, block
`[1, rep_n, rep_k, 2, 256] = [1, 2, 1, 2, 256]` (line 400). -/
def bScaleDesc (d : Inst) : TensorDesc .u8 :=
  ⟨d.b_scale.ptr 0, [1, d.N / 128, d.K / 128, 2, 256],
    [d.N / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 2, 1, 2, 256], .zero⟩

/-- `c_desc` (line 256): `TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])`: shape
`[M, N]`, strides `[N, 1]`, block `[128, 256]`, padding `"zero"`. -/
def cDesc (d : Inst) : TensorDesc .f16 := ⟨d.c.ptr 0, [d.M, d.N], [d.N, 1], [128, 256], .zero⟩

/-- The input bytes: the matrices `a`, `b` and the scale tensors. -/
def inEl (d : Inst) : Set Int :=
  d.a.elems {e | e < d.M * d.K} ∪ d.b.elems {e | e < d.N * d.K} ∪
    d.a_scale.elems {e | e < d.M / 128 * (d.K / 128) * 512} ∪
    d.b_scale.elems {e | e < d.N / 128 * (d.K / 128) * 512}

/-- The output bytes: the matrix `c`. -/
def cEl (d : Inst) : Set Int := d.c.elems {e | e < d.M * d.N}

/-- The instances covered. The sizes are variables; the conditions are those of the host's
tensors and the bounds the kernel's `int32` arithmetic and descriptors need, with `K ≤ 2^14`
for the tolerance. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M` is a positive multiple of `128` below `2^31`: the scales of `a` come in blocks of `128`
  rows (`M // 128`, line 370; the reference of line 416 needs them all), and `M` is a descriptor
  shape entry (`int32`). Every tile of `a` is then full. -/
  M_ok : 0 < d.M ∧ d.M % 128 = 0 ∧ d.M < 2 ^ 31
  /-- `N` is a positive multiple of `128` below `2^31` (`N // 128`, line 371). The last column of
  tiles may be half outside `c` (`BLOCK_N = 256`). -/
  N_ok : 0 < d.N ∧ d.N % 128 = 0 ∧ d.N < 2 ^ 31
  /-- `K` is a positive multiple of `128` (`K // VEC_SIZE // 4`, line 370, and the benchmark's
  `assert K % 128 == 0`, line 473), at most `2^14` (the module docstring's `mxfp8` benchmark
  goes up to `16384`; beyond, the tolerance grows: README, Narrowings). -/
  K_ok : 0 < d.K ∧ d.K % 128 = 0 ∧ d.K ≤ 2 ^ 14
  /-- The grid `cdiv(M, 128) · cdiv(N, 256)` (line 258) has fewer than `2^31` programs. -/
  grid_lt : (d.M + 127) / 128 * ((d.N + 255) / 256) < 2 ^ 31
  /-- Each tensor starts at a `16`-byte-aligned address (`TensorDesc.Valid`; PyTorch's
  allocations are). -/
  base_al : d.a.base % 16 = 0 ∧ d.b.base % 16 = 0 ∧ d.a_scale.base % 16 = 0 ∧
    d.b_scale.base % 16 = 0 ∧ d.c.base % 16 = 0
  /-- `a`'s storage holds `M · K` elements. -/
  a_len : d.M * d.K ≤ d.a.len
  /-- `b`'s storage holds `N · K` elements. -/
  b_len : d.N * d.K ≤ d.b.len
  /-- `a_scale`'s storage holds `(M / 128) · (K / 128) · 512` bytes. -/
  as_len : d.M / 128 * (d.K / 128) * 512 ≤ d.a_scale.len
  /-- `b_scale`'s storage holds `(N / 128) · (K / 128) · 512` bytes. -/
  bs_len : d.N / 128 * (d.K / 128) * 512 ≤ d.b_scale.len
  /-- `c`'s storage holds `M · N` elements. -/
  c_len : d.M * d.N ≤ d.c.len
  /-- The output shares no byte with an input (`torch.empty` of line 244 is a fresh tensor). -/
  c_in : Disjoint (cEl d) (inEl d)

/-- The model on an instance: the host's descriptors and the constexprs `block_scaled_matmul`
passes (lines 259–280) for `"mxfp8"` on the H200 (`SETUP.md`): `M`, `K`, `VEC_SIZE = 32`,
`BLOCK_M = 128`, `BLOCK_N = 256`, `BLOCK_K = 128`, `rep_m = 1`, `rep_n = 2`, `rep_k = 1`. -/
def model (d : Inst) : Kernel :=
  block_scaled_matmul_kernel (aDesc d) (aScaleDesc d) (bDesc d) (bScaleDesc d) (cDesc d)
    d.M d.K 32 128 256 128 1 2 1 rfl rfl

/-- The host's launch (lines 258–259): a grid of `cdiv(M, 128) · cdiv(N, 256)` programs (the
tuple `(…, 1)`); the source's block shapes: `[128, 256]` (the accumulator, `b.T`, the converted
result and `c`'s block), `[128, 128]` (`a`), `[256, 128]` (`b`), and for the scales of `a`
`[1, 1, 1, 2, 256]`, `[1, 1, 32, 4, 4]`, `[1, 4, 32, 1, 4]`, `[128, 4]`, of `b`
`[1, 2, 1, 2, 256]`, `[2, 1, 32, 4, 4]`, `[2, 4, 32, 1, 4]`, `[256, 4]`; `enable_fp_fusion` is
Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False` and
`debug = false`. The launch gives no other option (the default `num_warps`, `num_stages`; the
positional `NUM_STAGES` and the keyword `disallow_acc_multi_buffer` are kernel arguments). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.M + 127) / 128 * ((d.N + 255) / 256))
    [[128, 256], [128, 128], [256, 128], [1, 1, 1, 2, 256], [1, 1, 32, 4, 4], [1, 4, 32, 1, 4],
      [128, 4], [1, 2, 1, 2, 256], [2, 1, 32, 4, 4], [2, 4, 32, 1, 4], [256, 4]] (fusion := true)

/-- The inputs covered: every buffer is valid (aligned, inside `[0, 2^64)`, every byte present).
Any values in `a`, `b` and the scales (NaN included). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.a_scale.Valid init ∧ d.b_scale.Valid init ∧ d.c.Valid init

/-! ## The specification -/

/-- The value of an `e8m0` scale byte `s`: `2^(s − 127)` (OCP MX). -/
def scaleV (s : U8) : ℚ := 2 ^ ((s.bits.toNat : Int) - 127)

/-- The value of an element `x` of `a` or `b` with its scale `s`: `x · 2^(s − 127)`. -/
def scaled (x : F8E4) (s : U8) : ℚ := x.toRat * scaleV s

/-- The element domain the library's fact `Rules.Num.DotScaledE4m3` covers: `x` finite (not
NaN), the scale byte in `1 … 254` (`e8m0Dom`: `0x00` and the NaN `0xFF` excluded), and the
scaled value `0` or a normal `bfloat16` magnitude (`scaledDomBF16`: in `[2^-126, maxVal]`). -/
def InDom (x : F8E4) (s : U8) : Prop :=
  x.isFinite = true ∧ Rules.Num.e8m0Dom s.bits.toNat ∧ Rules.Num.scaledDomBF16 (scaled x s)

/-- Where the scale of element `(r, k)` of `a` (or `b`, `r` its row) lies in the scale tensor
of a matrix with `K` columns: entry `[r // 128, k // 128, r % 32, 4 · ((r % 128) // 32) + (k % 128) // 32]`
of the `[R // 128, K // 128, 32, 16]` layout (the 128 × 4 blocks of the cuBLAS/`tcgen05` layout,
lines 47–63), as the host's reference unpacks it (`unpack_scale`, lines 411–416: the permutation
`(0, 3, 2, 1, 4)` of the `[R // 128, K // 128, 32, 4, 4]` view). -/
def sIdx (K r k : Nat) : Nat :=
  (r / 128 * (K / 128) + k / 128) * 512 + 16 * (r % 32) + 4 * (r % 128 / 32) + k % 128 / 32

/-- The products of the reduction for one output element, from its row of `a` (`xa k`, scales
`sa k`) and its row of `b` (`xb k`, scales `sb k`): `scaled (xa k) (sa k) · scaled (xb k) (sb k)`. -/
def prodV (xa xb : Nat → F8E4) (sa sb : Nat → U8) (k : Nat) : ℚ :=
  scaled (xa k) (sa k) * scaled (xb k) (sb k)

/-- The exact sum of the first `K` products. -/
def dotSum (p : Nat → ℚ) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, p k

/-- The sum of their magnitudes. -/
def dotAbs (p : Nat → ℚ) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, |p k|

/-- The exact partial sum after `t` steps of the loop (the products `k < 128 t`). -/
def partSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (128 * t), p e

/-- The magnitude of step `t`'s products (`128 t ≤ k < 128 (t + 1)`). -/
def chunkAbs (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range 128, |p (128 * t + e)|

/-- The relative constant of one `tl.dot_scaled` step of `128` products: the library's
`relDot16 128 = 130 · 2^-22` (`Rules.Num.DotScaledE4m3`), about `3.10 · 10^-5`. -/
def rho : ℚ := Rules.Num.relDot16 128

/-- The absolute constant of one step: `absDot16 128 = 130 · 2^-126`. -/
def alpha : ℚ := Rules.Num.absDot16 128

/-- **The tolerance** after `t` steps, computed from the exact inputs: `tol p 0 = 0`, and step
`t` adds `rho · (|partSum p t| + tol p t + chunkAbs p t) + alpha`, the library's bound for one
`tl.dot_scaled` of `128` products into the accumulator, with the accumulator's magnitude bounded
by the exact partial sum plus the tolerance so far. The output's tolerance is `tol p (K / 128)`. -/
def tol (p : Nat → ℚ) : Nat → ℚ
  | 0 => 0
  | t + 1 => tol p t + rho * (|partSum p t| + tol p t + chunkAbs p t) + alpha

/-- What the launch leaves in element `(i, j)` of `c`: some `float16` value `c`; and when the
row `i` of `a` holds `xa k` with scales `sa k` (at `sIdx K i k` of `a_scale`) and the row `j` of
`b` holds `xb k` with scales `sb k` (`k < K`) in the initial memory, every element is in the
fact's domain (`InDom`), and the products' magnitudes plus the tolerance stay below `2^127`
(room in `float32`), then `c` is the `float16` rounding (to nearest even, up to NaN payload) of a
finite `float32` value `v` within `tol p (K / 128)` of the exact sum `S = Σ_k p k`
(`p = prodV xa xb sa sb`); and when there is room in `float16` (`|S| + tol ≤ 65504`), `c` is
finite and within `tol` plus one `float16` rounding (`(|S| + tol)·2^-11 + 2^-25`) of `S`. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.N + j)).addr = some c ∧
    ∀ (xa xb : Nat → F8E4) (sa sb : Nat → U8),
      (∀ k, k < d.K → init.read .f8e4nv (d.a.ptr ↑(i * d.K + k)).addr = some (xa k) ∧
        init.read .u8 (d.a_scale.ptr ↑(sIdx d.K i k)).addr = some (sa k)) →
      (∀ k, k < d.K → init.read .f8e4nv (d.b.ptr ↑(j * d.K + k)).addr = some (xb k) ∧
        init.read .u8 (d.b_scale.ptr ↑(sIdx d.K j k)).addr = some (sb k)) →
      (∀ k, k < d.K → InDom (xa k) (sa k) ∧ InDom (xb k) (sb k)) →
      dotAbs (prodV xa xb sa sb) d.K + tol (prodV xa xb sa sb) (d.K / 128) ≤ 2 ^ 127 →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - dotSum (prodV xa xb sa sb) d.K| ≤ tol (prodV xa xb sa sb) (d.K / 128) ∧
        FloatV.NaNEq (FloatV.cvt v) c ∧
        (|dotSum (prodV xa xb sa sb) d.K| + tol (prodV xa xb sa sb) (d.K / 128) ≤
            Math.Float.Format.binary16.maxVal →
          c.isFinite = true ∧
          |c.toRat - dotSum (prodV xa xb sa sb) d.K| ≤ tol (prodV xa xb sa sb) (d.K / 128) +
            (|dotSum (prodV xa xb sa sb) d.K| + tol (prodV xa xb sa sb) (d.K / 128)) *
              Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta)

/-- What the launch leaves: every element of `c` as `OutSpec` says; every byte outside the
matrix `c` unchanged (the inputs and any other byte of `init`). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i j, i < d.M → j < d.N → OutSpec d init fin i j) ∧ ∀ z, z ∉ cEl d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotScaledE4m3`: `tl.dot_scaled` of `"e4m3"` by `"e4m3"` with `e8m0` scales in
  groups of `32`, `K`-packed operands, either `fast_math` (the model's line 234): each output
  element whose row of `lhs` and column of `rhs` have finite scaled values in `scaledDomBF16`,
  scale bits in `e8m0Dom` and a finite accumulator is finite and within
  `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of `acc + Σ a·b`, when that leaves room in `float32`. -/
  dot : Rules.Num.DotScaledE4m3 P
  /-- [rule] `Num.DotScaledTotal`: the model's `tl.dot_scaled` (line 234) always has a result. -/
  dot_total : Rules.Num.DotScaledTotal P
    { lhs := .e4m3, rhs := .e4m3, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
      lhsKPack := true, rhsKPack := true, fastMath := false }

end Kernels.BlockScaledMxfp8
