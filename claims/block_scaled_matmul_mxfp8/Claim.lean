import TVA

/-!
# `block_scaled_matmul_mxfp8`: the prepared claim

Source: `source/10-block-scaled-matmul.py` (Triton 3.8.0's tutorial 10, unchanged; `SETUP.md`).
The kernel is `block_scaled_matmul_kernel` (lines 174–240); it reads the module-level names
`triton` (its decorator) and `tl` (lines 123–124). The host function `block_scaled_matmul` (lines
243–281) launches it with the config `initialize_block_scaled` (lines 330–449) builds for
`block_scale_type = "mxfp8"` or `"mixed"` on the H200: a `float8e4nv` operand `a` (`M × K`), an
operand `b` (`N × K`) of `float8e4nv` (`"mxfp8"`) or of `e2m1` packed two per byte along `K`
(`"mixed"`), `e8m0` scales (`uint8`) in groups of `VEC_SIZE = 32` along `K` in the 5-D
preshuffled layout, and a `float16` output `c` (`M × N`). The launches to cover are `SETUP.md`'s.

This file holds the model (`block_scaled_matmul_kernel`, `model`, `launch`), the conditions
(`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`, with the tolerance `tol`), the
assumptions (`Assumptions`) and the claim `Statement`, which it does not prove. It proves
`consistent`, `satisfiable`, and that each launch of `SETUP.md` is covered (`setup_*_ok`). See
`README.md`.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## The model -/

/-- The `tl.dot_scaled` configuration of line 230 (the `MIXED_PREC` branch): `"e4m3"` by
`"e2m1"`, `e8m0` scales on both sides (`uint8` tensors), `32` elements per scale (the dot's
`K = BLOCK_K = 128` over the scales' last dimension `BLOCK_K // VEC_SIZE = 4`), the defaults
`lhs_k_pack=True`, `rhs_k_pack=True` (the `e2m1` `rhs` holds two elements per byte along `K`) and
`fast_math=False`. -/
def cfgMixed : ScaledCfg :=
  { lhs := .e4m3, rhs := .e2m1, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
    lhsKPack := true, rhsKPack := true, fastMath := false }

/-- The configuration of line 232: `"e2m1"` by `"e2m1"`, otherwise as `cfgMixed`. No covered
instance reaches it (`ELEM_PER_BYTE_A = 1` on every instance). -/
def cfgFp4 : ScaledCfg :=
  { lhs := .e2m1, rhs := .e2m1, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
    lhsKPack := true, rhsKPack := true, fastMath := false }

/-- The configuration of line 234: `"e4m3"` by `"e4m3"`, otherwise as `cfgMixed`. -/
def cfgMx : ScaledCfg :=
  { lhs := .e4m3, rhs := .e4m3, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
    lhsKPack := true, rhsKPack := true, fastMath := false }

/-- `block_scaled_matmul_kernel` (source lines 174–240). The constexpr `output_type = 1` (the
value `block_scaled_matmul` passes for `dtype_dst = torch.float16`, lines 247–248, 269) is fixed,
so it is no parameter of the model: lines 197–202 give `output_dtype` the dtype `tl.float16` (a
compile-time value, kept as it is). `ELEM_PER_BYTE_A` and `ELEM_PER_BYTE_B` are Lean parameters:
line 216's `MIXED_PREC` is the compile-time value of `ELEM_PER_BYTE_A == 1 and
ELEM_PER_BYTE_B == 2`, and lines 229–234 are a compile-time `if`/`elif`/`else` on it and on
`ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2`, written as a Lean `if` on the parameters.
`NUM_STAGES` and `disallow_acc_multi_buffer` are only `tl.range` scheduling keywords (lines
219–220), and `N` is read by no line of the body, so they are no parameters either. The other
constexprs (`M`, `K`, `VEC_SIZE`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `rep_m`, `rep_n`, `rep_k`) are
Lean parameters; `model` passes the host's values.

The descriptor parameters receive the host's `TensorDescriptor`s (`TensorDesc`, with the host's
shape, strides, block shape and padding): `a_desc` and `b_desc` of the dtypes `da` and `db` of
the host's tensors (`float8e4nv` for an `e4m3` matrix, `uint8` for a packed `e2m1` one),
`a_scale_desc`, `b_scale_desc` of `uint8` (`e8m0` scales), `c_desc` of `float16`. Each
`desc.load` has the descriptor's block shape, written as the host's (lines 367–368, 399–400;
`block_scaled_matmul`'s line 256 for `c`).

`hA` and `hB`: the reshapes of lines 226–227 to `[BLOCK_M, BLOCK_K // VEC_SIZE]` and
`[BLOCK_N, BLOCK_K // VEC_SIZE]` keep the number of elements of the permuted scale blocks
(`[rep_m, 4, 32, rep_k, 4]` and `[rep_n, 4, 32, rep_k, 4]`); Triton rejects the kernel on the
host otherwise (`tl.reshape`, SEMANTICS.md, shape changes), so the model is defined for the
configurations where they hold (on every covered launch: `512 = 512` and `1024 = 1024`). -/
def block_scaled_matmul_kernel {da db : DType} (a_desc : TensorDesc da) (a_scale_desc : TensorDesc .u8)
    (b_desc : TensorDesc db) (b_scale_desc : TensorDesc .u8) (c_desc : TensorDesc .f16)
    (M K ELEM_PER_BYTE_A ELEM_PER_BYTE_B VEC_SIZE BLOCK_M BLOCK_N BLOCK_K rep_m rep_n rep_k : Nat)
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
    -- line 216: the annotated `MIXED_PREC` is a compile-time `bool` (see the docstring); its only
    -- use is line 229
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
      -- line 221: `a_desc`'s block `[BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A]` at the `int32` offsets;
      -- lanes outside the tensor hold the padding (`"zero"`)
      let a ← tl.desc_load (bs := [BLOCK_M, BLOCK_K / ELEM_PER_BYTE_A]) a_desc [offs_am, offs_k_a]
      -- line 222: `b_desc`'s block `[BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B]`
      let b ← tl.desc_load (bs := [BLOCK_N, BLOCK_K / ELEM_PER_BYTE_B]) b_desc [offs_bn, offs_k_b]
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
      -- lines 229–234: a compile-time `if`/`elif`/`else`; each `tl.dot_scaled` has `e8m0` scales
      -- (`uint8`), group `BLOCK_K / (BLOCK_K // VEC_SIZE) = 32`, `K`-packed operands, the default
      -- `fast_math=False`; `b.T` is `b.trans`
      let accumulator ←
        if ELEM_PER_BYTE_A = 1 ∧ ELEM_PER_BYTE_B = 2 then
          -- line 230: `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)`
          tl.dot_scaled cfgMixed BLOCK_M BLOCK_N BLOCK_K a (some scale_a) b.trans (some scale_b) accumulator
        else if ELEM_PER_BYTE_A = 2 ∧ ELEM_PER_BYTE_B = 2 then
          -- line 232: `tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)`
          tl.dot_scaled cfgFp4 BLOCK_M BLOCK_N BLOCK_K a (some scale_a) b.trans (some scale_b) accumulator
        else
          -- line 234: `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)`
          tl.dot_scaled cfgMx BLOCK_M BLOCK_N BLOCK_K a (some scale_a) b.trans (some scale_b) accumulator
      -- lines 236–237: `BLOCK_K // ELEM_PER_BYTE_*` of two constexprs is Python's value, built in
      -- `int32`; the wrapping `int32` sum
      let offs_k_a := offs_k_a + (IntV.ofNat (BLOCK_K / ELEM_PER_BYTE_A) : I32)
      let offs_k_b := offs_k_b + (IntV.ofNat (BLOCK_K / ELEM_PER_BYTE_B) : I32)
      -- line 238: the constexpr `rep_k` built in `int32`
      let offs_scale_k := offs_scale_k + (IntV.ofNat rep_k : I32)
      return (accumulator, offs_k_a, offs_k_b, offs_scale_k)
    -- line 240: `accumulator.to(output_dtype)` with `output_dtype = tl.float16`, one rounding to
    -- nearest even (`tl.cvt`), evaluated before the store; `c_desc`'s block at `[offs_am, offs_bn]`;
    -- lanes outside the tensor are not stored
    let c : Tensor [BLOCK_M, BLOCK_N] F16 ← tl.cvt accumulator
    tl.desc_store c_desc [offs_am, offs_bn] c

/-! ## The conditions -/

/-- The storage dtype of `b`: `uint8` (two `e2m1` elements per byte, `to_packed_tensor(dim=1)`,
line 363) for `"mixed"`, `float8e4nv` (line 361) for `"mxfp8"`. -/
def bDT : Bool → DType
  | true => .u8
  | false => .f8e4nv

/-- `ELEM_PER_BYTE_B` (line 337): `2` for `"mixed"`, `1` for `"mxfp8"`. -/
def epbB (mixed : Bool) : Nat := if mixed then 2 else 1

/-- What the host chooses: one launch of `block_scaled_matmul_kernel` through
`block_scaled_matmul(…, torch.float16, M, N, K, …)` on the tensors `initialize_block_scaled(M, N,
K, block_scale_type)` builds (lines 340–404) for `block_scale_type = "mixed"` (`mixed = true`) or
`"mxfp8"` (`mixed = false`), all contiguous:
* `a`: `M × K` `float8e4nv` (`a_ref.to(torch.float8_e4m3fn)`, line 354), element `(i, k)` at
  `i * K + k`;
* `b`: for `"mxfp8"`, `N × K` `float8e4nv` (line 361), element `(j, k)` at `j * K + k`; for
  `"mixed"`, `N × (K / 2)` `uint8` (`b_ref.to_packed_tensor(dim=1)`, line 363), element `(j, k)`
  in byte `j * (K / 2) + k / 2`, the low nibble for an even `k`, the high one for an odd `k`;
* `a_scale`: the `uint8` `e8m0` scales of `a` (`MXScaleTensor(…).data`, lines 387–389), shape
  `[M // 128, K // 128, 32, 16]` (line 370), viewed as `[1, M // 128, K // 128, 2, 256]`
  (line 401);
* `b_scale`: those of `b`, `[N // 128, K // 128, 32, 16]` (line 371), viewed as
  `[1, N // 128, K // 128, 2, 256]` (line 402);
* `c`: the `M × N` `float16` output (`torch.empty((M, N), dtype=torch.float16)`, line 244),
  element `(i, j)` at `i * N + j`. -/
structure Inst where
  /-- `block_scale_type == "mixed"` (`ELEM_PER_BYTE_B = 2`, line 337); `false` for `"mxfp8"`. -/
  mixed : Bool
  /-- `M`: the rows of `a` and `c` (a constexpr). -/
  M : Nat
  /-- `N`: the rows of `b`, the columns of `c` (a constexpr). -/
  N : Nat
  /-- `K`: the reduction length, in elements (a constexpr). -/
  K : Nat
  /-- The storage of `a`. -/
  a : Buf .f8e4nv
  /-- The storage of `b` (`bDT mixed`). -/
  b : Buf (bDT mixed)
  /-- The storage of `a`'s scales. -/
  a_scale : Buf .u8
  /-- The storage of `b`'s scales. -/
  b_scale : Buf .u8
  /-- The storage of the output `c`. -/
  c : Buf .f16

/-- `a_desc` (line 367): `TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])`:
shape `[M, K]`, strides `[K, 1]`, block `[128, 128]`, padding `"zero"`. -/
def aDesc (d : Inst) : TensorDesc .f8e4nv := ⟨d.a.ptr 0, [d.M, d.K], [d.K, 1], [128, 128], .zero⟩

/-- `b_desc` (line 368): `TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])`:
shape `[N, K / e]`, strides `[K / e, 1]`, block `[256, 128 / e]` (`e = ELEM_PER_BYTE_B`: `[256, 128]`
for `"mxfp8"`, `[256, 64]` for `"mixed"`), padding `"zero"`. -/
def bDesc (d : Inst) : TensorDesc (bDT d.mixed) :=
  ⟨d.b.ptr 0, [d.N, d.K / epbB d.mixed], [d.K / epbB d.mixed, 1], [256, 128 / epbB d.mixed], .zero⟩

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
  d.a.elems {e | e < d.M * d.K} ∪ d.b.elems {e | e < d.N * (d.K / epbB d.mixed)} ∪
    d.a_scale.elems {e | e < d.M / 128 * (d.K / 128) * 512} ∪
    d.b_scale.elems {e | e < d.N / 128 * (d.K / 128) * 512}

/-- The output bytes: the matrix `c`. -/
def cEl (d : Inst) : Set Int := d.c.elems {e | e < d.M * d.N}

/-- The instances covered. The sizes are variables; the conditions are those of the host's
tensors and the bounds the kernel's `int32` arithmetic and descriptors need, with `K ≤ 2^17`
so that the tolerance stays meaningful (README, Exclusions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `M` is a positive multiple of `128` below `2^31`: the scales of `a` come in blocks of `128`
  rows (`M // 128`, line 370; the reference of line 416 needs them all), and `M` is a descriptor
  shape entry (`int32`). Every tile of `a` is then full. -/
  M_ok : 0 < d.M ∧ d.M % 128 = 0 ∧ d.M < 2 ^ 31
  /-- `N` is a positive multiple of `128` below `2^31` (`N // 128`, line 371). The last column of
  tiles may be half outside `c` (`BLOCK_N = 256`). -/
  N_ok : 0 < d.N ∧ d.N % 128 = 0 ∧ d.N < 2 ^ 31
  /-- `K` is a positive multiple of `128` (`K // VEC_SIZE // 4`, line 370, and the benchmark's
  `assert K % 128 == 0`, line 473), at most `2^17` (the module docstring's `mxfp8` benchmark
  goes up to `16384`, the test uses `8192`; `2^17` is eight times the largest; beyond it the
  tolerance grows past a few percent of the typical output: README, Exclusions). -/
  K_ok : 0 < d.K ∧ d.K % 128 = 0 ∧ d.K ≤ 2 ^ 17
  /-- The grid `cdiv(M, 128) · cdiv(N, 256)` (line 258) has fewer than `2^31` programs. -/
  grid_lt : (d.M + 127) / 128 * ((d.N + 255) / 256) < 2 ^ 31
  /-- Each tensor starts at a `16`-byte-aligned address (`TensorDesc.Valid`; PyTorch's
  allocations are). -/
  base_al : d.a.base % 16 = 0 ∧ d.b.base % 16 = 0 ∧ d.a_scale.base % 16 = 0 ∧
    d.b_scale.base % 16 = 0 ∧ d.c.base % 16 = 0
  /-- `a`'s storage holds `M · K` elements. -/
  a_len : d.M * d.K ≤ d.a.len
  /-- `b`'s storage holds `N · (K / ELEM_PER_BYTE_B)` elements (`N · K` for `"mxfp8"`,
  `N · K / 2` bytes for `"mixed"`). -/
  b_len : d.N * (d.K / epbB d.mixed) ≤ d.b.len
  /-- `a_scale`'s storage holds `(M / 128) · (K / 128) · 512` bytes. -/
  as_len : d.M / 128 * (d.K / 128) * 512 ≤ d.a_scale.len
  /-- `b_scale`'s storage holds `(N / 128) · (K / 128) · 512` bytes. -/
  bs_len : d.N / 128 * (d.K / 128) * 512 ≤ d.b_scale.len
  /-- `c`'s storage holds `M · N` elements. -/
  c_len : d.M * d.N ≤ d.c.len
  /-- The output shares no byte with an input (`torch.empty` of line 244 is a fresh tensor). -/
  c_in : Disjoint (cEl d) (inEl d)
  /-- `M` counts tokens (the rows of the activations `a`): a context length, at most `2^22`. -/
  M_ctx : d.M ≤ 2 ^ 22
  /-- `N` is a model dimension (the output features): a positive multiple of `128` (`N_ok`), at
  most `2^17`. (`K`, the input features, is one by `K_ok`.) -/
  N_dim : d.N ≤ 2 ^ 17

/-- The model on an instance: the host's descriptors and the constexprs `block_scaled_matmul`
passes (lines 259–280) for `"mxfp8"` or `"mixed"` on the H200 (`SETUP.md`): `M`, `K`,
`ELEM_PER_BYTE_A = 1`, `ELEM_PER_BYTE_B = epbB mixed`, `VEC_SIZE = 32`, `BLOCK_M = 128`,
`BLOCK_N = 256`, `BLOCK_K = 128`, `rep_m = 1`, `rep_n = 2`, `rep_k = 1`. -/
def model (d : Inst) : Kernel :=
  block_scaled_matmul_kernel (aDesc d) (aScaleDesc d) (bDesc d) (bScaleDesc d) (cDesc d)
    d.M d.K 1 (epbB d.mixed) 32 128 256 128 1 2 1 rfl rfl

/-- The host's launch (lines 258–259): a grid of `cdiv(M, 128) · cdiv(N, 256)` programs (the
tuple `(…, 1)`); the source's block shapes: `[128, 256]` (the accumulator, the converted result
and `c`'s block), `[128, 128]` (`a`), `b`'s block and `b.T`'s (`[256, 128]` and `[128, 256]` for
`"mxfp8"`, `[256, 64]` and `[64, 256]` for `"mixed"`), and for the scales of `a`
`[1, 1, 1, 2, 256]`, `[1, 1, 32, 4, 4]`, `[1, 4, 32, 1, 4]`, `[128, 4]`, of `b`
`[1, 2, 1, 2, 256]`, `[2, 1, 32, 4, 4]`, `[2, 4, 32, 1, 4]`, `[256, 4]`; `enable_fp_fusion` is
Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False` and
`debug = false`. The launch gives no other option (the default `num_warps`, `num_stages`; the
positional `NUM_STAGES` and the keyword `disallow_acc_multi_buffer` are kernel arguments). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.M + 127) / 128 * ((d.N + 255) / 256))
    [[128, 256], [128, 128], [256, 128 / epbB d.mixed], [128 / epbB d.mixed, 256], [1, 1, 1, 2, 256],
      [1, 1, 32, 4, 4], [1, 4, 32, 1, 4], [128, 4], [1, 2, 1, 2, 256], [2, 1, 32, 4, 4],
      [2, 4, 32, 1, 4], [256, 4]] (fusion := true)

/-! ## The inputs' values and the reference -/

/-- The value of an `e8m0` scale byte `s`: `2^(s − 127)` (OCP MX). -/
def scaleV (s : U8) : ℚ := 2 ^ ((s.bits.toNat : Int) - 127)

/-- The value of an element of value `x` with its scale `s`: `x · 2^(s − 127)`. -/
def scaled (x : ℚ) (s : U8) : ℚ := x * scaleV s

/-- The element domain the library's facts `Rules.Num.DotScaledE4m3` and
`Rules.Num.DotScaledE4m3E2m1` cover, for an element whose bits are finite (`fin`) with value `x`
and scale byte `s`: the scale byte in `1 … 254` (`e8m0Dom`: `0x00` and the NaN `0xFF` excluded),
and the scaled value `0` or a normal `bfloat16` magnitude (`scaledDomBF16`: in `[2^-126, maxVal]`). -/
def InDom (fin : Bool) (x : ℚ) (s : U8) : Prop :=
  fin = true ∧ Rules.Num.e8m0Dom s.bits.toNat ∧ Rules.Num.scaledDomBF16 (scaled x s)

/-- Where the scale of element `(r, k)` of `a` (or `b`, `r` its row) lies in the scale tensor
of a matrix with `K` columns: entry `[r // 128, k // 128, r % 32, 4 · ((r % 128) // 32) + (k % 128) // 32]`
of the `[R // 128, K // 128, 32, 16]` layout (the 128 × 4 blocks of the cuBLAS/`tcgen05` layout,
lines 47–63), as the host's reference unpacks it (`unpack_scale`, lines 411–416: the permutation
`(0, 3, 2, 1, 4)` of the `[R // 128, K // 128, 32, 4, 4]` view). -/
def sIdx (K r k : Nat) : Nat :=
  (r / 128 * (K / 128) + k / 128) * 512 + 16 * (r % 32) + 4 * (r % 128 / 32) + k % 128 / 32

/-- The products of the reduction for one output element, from the values of its row of `a`
(`va k`, scales `sa k`) and of its row of `b` (`vb k`, scales `sb k`):
`scaled (va k) (sa k) · scaled (vb k) (sb k)`. -/
def prodV (va vb : Nat → ℚ) (sa sb : Nat → U8) (k : Nat) : ℚ :=
  scaled (va k) (sa k) * scaled (vb k) (sb k)

/-- The exact sum of the first `K` products. -/
def dotSum (p : Nat → ℚ) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, p k

/-- The sum of their magnitudes. -/
def dotAbs (p : Nat → ℚ) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, |p k|

/-- The exact partial sum after `t` steps of the loop (the products `k < 128 t`). -/
def partSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (128 * t), p e

/-- The magnitude of step `t`'s products (`128 t ≤ k < 128 (t + 1)`). -/
def chunkAbs (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range 128, |p (128 * t + e)|

/-- The relative constant of one `tl.dot_scaled` step of `128` products: the library's
`relDot16 128 = 130 · 2^-22` (`Rules.Num.DotScaledE4m3` and `Rules.Num.DotScaledE4m3E2m1`), about
`3.10 · 10^-5`. -/
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

/-- The value element `i` of a `float8e4nv` buffer holds in memory `m` (`+0` if it cannot be
read; `Pre` makes every buffer valid, so every element the claim reads can). -/
def rd8 (m : Mem) (b : Buf .f8e4nv) (i : Nat) : F8E4 :=
  (m.read .f8e4nv (b.ptr (i : Int)).addr).getD (FloatV.ofPat 0)

/-- The byte element `i` of a `uint8` buffer holds in memory `m` (`0` if it cannot be read). -/
def rdU8 (m : Mem) (b : Buf .u8) (i : Nat) : U8 :=
  (m.read .u8 (b.ptr (i : Int)).addr).getD (IntV.ofNat 0)

/-- The bit pattern element `i` of a buffer holds in memory `m` (`0` if it cannot be read). -/
def rdBits {dt : DType} (m : Mem) (b : Buf dt) (i : Nat) : Nat :=
  ((m.read dt (b.ptr (i : Int)).addr).map dt.toBits).getD 0

/-- Element `(i, k)` of `a` (element `i·K + k` of its storage) in memory `m`. -/
def xa (d : Inst) (m : Mem) (i k : Nat) : F8E4 := rd8 m d.a (i * d.K + k)

/-- The element format of `b`: `e2m1` for `"mixed"`, `e4m3` for `"mxfp8"`. -/
def bFmt (mixed : Bool) : Math.Float.Format :=
  if mixed then Math.Float.Format.e2m1 else Math.Float.Format.e4m3

/-- The bits of element `(j, k)` of `b`: for `"mxfp8"` the byte `j·K + k`; for `"mixed"` the
nibble `k % 2` (low first: `(high << 4) | low`, `MXFP4Tensor.to_packed_tensor`) of byte
`j·(K / 2) + k / 2`. -/
def xbBits (d : Inst) (m : Mem) (j k : Nat) : Nat :=
  if d.mixed then (rdBits m d.b (j * (d.K / 2) + k / 2) >>> (4 * (k % 2))) % 16
  else rdBits m d.b (j * d.K + k)

/-- Whether element `(j, k)` of `b` is finite (every `e2m1` pattern is). -/
def bFin (d : Inst) (m : Mem) (j k : Nat) : Bool := (bFmt d.mixed).isFinite (xbBits d m j k)

/-- The value of element `(j, k)` of `b`. -/
def bVal (d : Inst) (m : Mem) (j k : Nat) : ℚ := (bFmt d.mixed).toRat (xbBits d m j k)

/-- The scale byte of element `(i, k)` of `a`: byte `sIdx K i k` of `a_scale`. -/
def sa (d : Inst) (m : Mem) (i k : Nat) : U8 := rdU8 m d.a_scale (sIdx d.K i k)

/-- The scale byte of element `(j, k)` of `b`: byte `sIdx K j k` of `b_scale`. -/
def sb (d : Inst) (m : Mem) (j k : Nat) : U8 := rdU8 m d.b_scale (sIdx d.K j k)

/-- The products of output element `(i, j)` in memory `m`:
`p k = scaled (xa i k) (sa i k) · scaled (bVal j k) (sb j k)` for `k < K`. -/
def prods (d : Inst) (m : Mem) (i j : Nat) : Nat → ℚ :=
  prodV (fun k => (xa d m i k).toRat) (bVal d m j) (sa d m i) (sb d m j)

/-- The exact result of element `(i, j)`: `S = Σ_{k<K} p k`. -/
def ref (d : Inst) (m : Mem) (i j : Nat) : ℚ := dotSum (prods d m i j) d.K

/-- Its tolerance: `tol p (K / 128)`, the recursion over the `K / 128` steps of the loop. -/
def tolOut (d : Inst) (m : Mem) (i j : Nat) : ℚ := tol (prods d m i j) (d.K / 128)

/-- The input values covered:
* every element of `a` and of `b` in the library facts' domain with its scale (`InDom`: finite,
  the scale byte in `1 … 254`, the scaled value `0` or a normal `bfloat16` magnitude);
* room in `float32` for every output element: `Σ_k |p k| + tol ≤ 2^127`. -/
structure ValsOk (d : Inst) (m : Mem) : Prop where
  /-- Every element of `a` with its scale is in the domain. -/
  a_dom : ∀ i k, i < d.M → k < d.K → InDom (xa d m i k).isFinite (xa d m i k).toRat (sa d m i k)
  /-- Every element of `b` with its scale is in the domain. -/
  b_dom : ∀ j k, j < d.N → k < d.K → InDom (bFin d m j k) (bVal d m j k) (sb d m j k)
  /-- Room in `float32` for every output element. -/
  room : ∀ i j, i < d.M → j < d.N → dotAbs (prods d m i j) d.K + tolOut d m i j ≤ 2 ^ 127

/-- The inputs covered: every buffer is valid (aligned, inside `[0, 2^64)`, every byte present),
and the input values are covered (`ValsOk`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.a_scale.Valid init ∧ d.b_scale.Valid init ∧ d.c.Valid init ∧
    ValsOk d init

/-! ## The specification -/

/-- What the launch leaves in element `(i, j)` of `c`: a `float16` value `c` that is the rounding
to nearest even (`FloatV.cvt`, up to NaN payload) of a finite `float32` value `v` within `tol` of
the exact result `S` (`ref`, from the initial memory); and when there is room in `float16`
(`|S| + tol ≤ 65504`), `c` is finite and within `tol` plus one `float16` rounding
(`(|S| + tol)·2^-11 + 2^-25`) of `S`. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : F16, fin.read .f16 (d.c.ptr ↑(i * d.N + j)).addr = some c ∧
    ∃ v : F32, v.isFinite = true ∧ |v.toRat - ref d init i j| ≤ tolOut d init i j ∧
      FloatV.NaNEq (FloatV.cvt v) c ∧
      (|ref d init i j| + tolOut d init i j ≤ Math.Float.Format.binary16.maxVal →
        c.isFinite = true ∧
        |c.toRat - ref d init i j| ≤ tolOut d init i j +
          (|ref d init i j| + tolOut d init i j) * Math.Float.Format.binary16.u +
            Math.Float.Format.binary16.eta)

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
  groups of `32`, `K`-packed operands, either `fast_math` (the model's line 234, `"mxfp8"`): each
  output element whose row of `lhs` and column of `rhs` have finite scaled values in
  `scaledDomBF16`, scale bits in `e8m0Dom` and a finite accumulator is finite and within
  `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of `acc + Σ a·b`, when that leaves room in `float32`. -/
  dot : Rules.Num.DotScaledE4m3 P
  /-- [rule] `Num.DotScaledE4m3E2m1`: the same for `"e4m3"` by `"e2m1"` (the model's line 230,
  `"mixed"`), with the same constants. -/
  dot_mixed : Rules.Num.DotScaledE4m3E2m1 P
  /-- [rule] `Num.DotScaledTotal`: the model's `tl.dot_scaled` of line 234 always has a result. -/
  dot_total : Rules.Num.DotScaledTotal P
    { lhs := .e4m3, rhs := .e4m3, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
      lhsKPack := true, rhsKPack := true, fastMath := false }
  /-- [rule] `Num.DotScaledTotal`: the model's `tl.dot_scaled` of line 230 always has a result. -/
  dot_total_mixed : Rules.Num.DotScaledTotal P
    { lhs := .e4m3, rhs := .e2m1, lhsScale := .e8m0, rhsScale := .e8m0, group := 32,
      lhsKPack := true, rhsKPack := true, fastMath := false }

/-! ## The claim -/

/-- The claim to prove: under the assumptions, the launch meets the specification on every
covered instance and input. Not proved here. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot,
    ⟨fun d _ => hf d, Platform.sm90ref.dotScaledE4m3, Platform.sm90ref.dotScaledE4m3E2m1,
      Platform.ref.dotScaledTotal _, Platform.ref.dotScaledTotal _⟩⟩

/-! ## Coverage: the launches of `SETUP.md`

`hostInst mixed M N K` is the instance `initialize_block_scaled(M, N, K, block_scale_type)` and
`block_scaled_matmul` build (`"mixed"` when `mixed`, else `"mxfp8"`): contiguous `a` (`M × K`),
`b` (`N × K`, or `N × K / 2` packed bytes), the scale tensors of `(M / 128)·(K / 128)·512` and
`(N / 128)·(K / 128)·512` bytes, and a fresh `c` (`M × N`). The five buffers sit at fixed,
aligned, far-apart addresses; the claim covers any addresses `Inst.ok` allows, and these only
witness the sizes. -/

/-- The host's instance for the format and the sizes `(M, N, K)`. -/
def hostInst (mixed : Bool) (M N K : Nat) : Inst where
  mixed := mixed
  M := M
  N := N
  K := K
  a := ⟨2 ^ 40, M * K⟩
  b := ⟨2 * 2 ^ 40, N * (K / epbB mixed)⟩
  a_scale := ⟨3 * 2 ^ 40, M / 128 * (K / 128) * 512⟩
  b_scale := ⟨4 * 2 ^ 40, N / 128 * (K / 128) * 512⟩
  c := ⟨5 * 2 ^ 40, M * N⟩

/-- Two buffers whose byte ranges are in order share no byte. -/
theorem bytes_disjoint {d₁ d₂ : DType} {x : Buf d₁} {y : Buf d₂}
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The first `n ≤ len` elements of a buffer lie in its bytes. -/
theorem elems_lt_sub {dt : DType} (b : Buf dt) {n : Nat} (hn : n ≤ b.len) :
    b.elems {e | e < n} ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hj => lt_of_lt_of_le hj hn

/-- `b`'s elements take one byte in either format. -/
theorem bDT_bytes (mixed : Bool) : (bDT mixed).bytes = 1 := by cases mixed <;> rfl

/-- `b`'s fields on the host's instance (its type depends on `mixed`, so they are stated apart). -/
theorem hostInst_b_base (mixed : Bool) (M N K : Nat) : (hostInst mixed M N K).b.base = 2 * 2 ^ 40 :=
  rfl
theorem hostInst_b_len (mixed : Bool) (M N K : Nat) :
    (hostInst mixed M N K).b.len = N * (K / epbB mixed) := rfl

/-- `K / ELEM_PER_BYTE_B ≤ K`. -/
theorem div_epbB_le (mixed : Bool) (K : Nat) : K / epbB mixed ≤ K := Nat.div_le_self _ _

/-- The host's instance is covered when its sizes meet `Inst.ok`'s bounds and fit the address
layout of `hostInst` (each tensor below `2^40` bytes). -/
theorem hostInst_ok {mixed : Bool} {M N K : Nat}
    (h : 0 < M ∧ M % 128 = 0 ∧ M < 2 ^ 31 ∧ 0 < N ∧ N % 128 = 0 ∧ N < 2 ^ 31 ∧
      0 < K ∧ K % 128 = 0 ∧ K ≤ 2 ^ 17 ∧ (M + 127) / 128 * ((N + 255) / 256) < 2 ^ 31 ∧
      M * K ≤ 2 ^ 40 ∧ N * K ≤ 2 ^ 40 ∧ M / 128 * (K / 128) * 512 ≤ 2 ^ 40 ∧
      N / 128 * (K / 128) * 512 ≤ 2 ^ 40 ∧ M * N * 2 ≤ 2 ^ 40 ∧ M ≤ 2 ^ 22 ∧ N ≤ 2 ^ 17) :
    (hostInst mixed M N K).ok := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, hA, hB, hSA, hSB, hC, hM22, hN17⟩ := h
  have hb : N * (K / epbB mixed) ≤ 2 ^ 40 :=
    le_trans (Nat.mul_le_mul_left N (div_epbB_le mixed K)) hB
  refine ⟨⟨h1, h2, h3⟩, ⟨h4, h5, h6⟩, ⟨h7, h8, h9⟩, h10, ?_, le_refl _, le_refl _, le_refl _,
    le_refl _, le_refl _, ?_, hM22, hN17⟩
  · simp only [hostInst_b_base]; simp only [hostInst]; omega
  · have hc : cEl (hostInst mixed M N K) ⊆ (hostInst mixed M N K).c.bytes :=
      elems_lt_sub _ (le_refl _)
    simp only [inEl]
    refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
    all_goals
      refine Disjoint.mono hc (elems_lt_sub _ (le_refl _)) (bytes_disjoint ?_)
      try rw [bDT_bytes]
      try simp only [hostInst_b_base, hostInst_b_len]
      simp only [hostInst, DType.bytes]
      omega

/-- The `"mxfp8"` test, `validate_block_scaled(8192, 8192, 8192, "mxfp8")` (source line 749). -/
def setup_test_8192 : Inst := hostInst false 8192 8192 8192
theorem setup_test_8192_ok : setup_test_8192.ok := hostInst_ok (by norm_num)

/-- The `"mxfp8"` benchmark at its default `K = 512` (`-K`, line 731): `M = N = 8192` (lines
474–475). -/
def setup_bench_k512 : Inst := hostInst false 8192 8192 512
theorem setup_bench_k512_ok : setup_bench_k512.ok := hostInst_ok (by norm_num)

/-- The module docstring's `mxfp8` benchmark, `--K_range 8192 16384 --K_step 2048`: `K = 8192`. -/
def setup_bench_k8192 : Inst := hostInst false 8192 8192 8192
theorem setup_bench_k8192_ok : setup_bench_k8192.ok := hostInst_ok (by norm_num)

/-- `K = 10240`. -/
def setup_bench_k10240 : Inst := hostInst false 8192 8192 10240
theorem setup_bench_k10240_ok : setup_bench_k10240.ok := hostInst_ok (by norm_num)

/-- `K = 12288`. -/
def setup_bench_k12288 : Inst := hostInst false 8192 8192 12288
theorem setup_bench_k12288_ok : setup_bench_k12288.ok := hostInst_ok (by norm_num)

/-- `K = 14336`. -/
def setup_bench_k14336 : Inst := hostInst false 8192 8192 14336
theorem setup_bench_k14336_ok : setup_bench_k14336.ok := hostInst_ok (by norm_num)

/-- `K = 16384`, the docstring's largest. -/
def setup_bench_k16384 : Inst := hostInst false 8192 8192 16384
theorem setup_bench_k16384_ok : setup_bench_k16384.ok := hostInst_ok (by norm_num)

/-- The `"mxfp8"` benchmark's sizes at the largest covered `K = 2^17` (any `-K` multiple of `128`
up to it is covered: `hostInst_ok`). -/
def setup_bench_k131072 : Inst := hostInst false 8192 8192 (2 ^ 17)
theorem setup_bench_k131072_ok : setup_bench_k131072.ok := hostInst_ok (by norm_num)

/-- The `"mixed"` test, `validate_block_scaled(8192, 8192, 8192, "mixed")` (line 749 with
`--format mixed`). -/
def setup_mixed_test_8192 : Inst := hostInst true 8192 8192 8192
theorem setup_mixed_test_8192_ok : setup_mixed_test_8192.ok := hostInst_ok (by norm_num)

/-- The `"mixed"` benchmark at the default `K = 512` (`--format mixed`, `-K` 512, line 731). -/
def setup_mixed_bench_k512 : Inst := hostInst true 8192 8192 512
theorem setup_mixed_bench_k512_ok : setup_mixed_bench_k512.ok := hostInst_ok (by norm_num)

/-- The `"mixed"` benchmark's sizes at the largest covered `K = 2^17` (any `-K` multiple of `128`
up to it is covered: `hostInst_ok`). -/
def setup_mixed_bench_k131072 : Inst := hostInst true 8192 8192 (2 ^ 17)
theorem setup_mixed_bench_k131072_ok : setup_mixed_bench_k131072.ok := hostInst_ok (by norm_num)

/-! ## Not vacuous: a covered instance and memory

The smallest covered sizes, `M = N = K = 128`, `"mxfp8"`, in a memory where every byte is
`0x38`: each element of `a` and `b` is the `e4m3` value `1.0` and each scale byte `56` is
`2^-71`, so every scaled value is `2^-71` (a normal `bfloat16` magnitude) and every product
`2^-142`. -/

/-- `M = N = K = 128`, `"mxfp8"`, the buffers back to back at 16-byte-aligned addresses. -/
def satInst : Inst where
  mixed := false
  M := 128
  N := 128
  K := 128
  a := ⟨0, 16384⟩
  b := ⟨16384, 16384⟩
  a_scale := ⟨32768, 512⟩
  b_scale := ⟨33280, 512⟩
  c := ⟨33792, 16384⟩

/-- `b`'s fields on `satInst`. -/
theorem satInst_b_base : satInst.b.base = 16384 := rfl
theorem satInst_b_len : satInst.b.len = 16384 := rfl

/-- Every byte present and `0x38`. -/
def satMem : Mem := fun _ => some 56

theorem satInst_ok : satInst.ok := by
  refine ⟨by norm_num [satInst], by norm_num [satInst], by norm_num [satInst], by norm_num [satInst],
    by norm_num [satInst, satInst_b_base], by norm_num [satInst],
    by norm_num [satInst, satInst_b_len, epbB], by norm_num [satInst],
    by norm_num [satInst], by norm_num [satInst], ?_, by norm_num [satInst], by norm_num [satInst]⟩
  have hc : cEl satInst ⊆ satInst.c.bytes := elems_lt_sub _ (by norm_num [satInst])
  simp only [inEl]
  refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
  all_goals
    refine Disjoint.mono hc (elems_lt_sub _ (by norm_num [satInst, satInst_b_len, epbB]))
      (bytes_disjoint ?_)
    try rw [bDT_bytes]
    try simp only [satInst_b_base, satInst_b_len]
    simp only [satInst, DType.bytes]
    omega

/-- A one-byte read of `satMem` at an address in `[0, 2^64)`. -/
theorem satMem_read1 {dt : DType} (hdt : dt.bytes = 1) {x : Int} (h0 : 0 ≤ x) (h1 : x + 1 ≤ 2 ^ 64) :
    satMem.read dt x = some (dt.ofBits (Math.Bytes.leValue [56])) := by
  have hal : Mem.Aligned dt x := ⟨h0, by rw [hdt]; exact h1, by rw [hdt]; exact Int.emod_one x⟩
  unfold Mem.read
  simp only [hal, ↓reduceIte, hdt]
  rfl

/-- The `e4m3` value of the byte `0x38`. -/
def one8 : F8E4 := DType.f8e4nv.ofBits (Math.Bytes.leValue [56])

/-- The `uint8` value of the byte `0x38`. -/
def b56 : U8 := DType.u8.ofBits (Math.Bytes.leValue [56])

theorem one8_fin : one8.isFinite = true := by decide

theorem one8_val : one8.toRat = 1 := by decide +kernel

theorem b56_bits : b56.bits.toNat = 56 := by decide +kernel

theorem sat_xa {i k : Nat} (hi : i < 128) (hk : k < 128) : xa satInst satMem i k = one8 := by
  simp only [xa, rd8, Buf.ptr_addr]
  rw [satMem_read1 rfl (by simp [satInst, DType.bytes]; omega) (by simp [satInst, DType.bytes]; omega)]
  rfl

theorem sat_xbBits {j k : Nat} (hj : j < 128) (hk : k < 128) : xbBits satInst satMem j k = 56 := by
  have hm : satInst.mixed = false := rfl
  simp only [xbBits, hm, Bool.false_eq_true, ↓reduceIte, rdBits, Buf.ptr_addr]
  rw [satMem_read1 rfl (by simp [satInst, DType.bytes, bDT]; omega)
    (by simp [satInst, DType.bytes, bDT]; omega)]
  decide +kernel

theorem sat_bFin {j k : Nat} (hj : j < 128) (hk : k < 128) : bFin satInst satMem j k = true := by
  rw [bFin, sat_xbBits hj hk]; decide +kernel

theorem sat_bVal {j k : Nat} (hj : j < 128) (hk : k < 128) : bVal satInst satMem j k = 1 := by
  rw [bVal, sat_xbBits hj hk]; decide +kernel

theorem sIdx_lt {r k : Nat} (hr : r < 128) (hk : k < 128) : sIdx 128 r k < 512 := by
  unfold sIdx; omega

theorem sat_sa {i k : Nat} (hi : i < 128) (hk : k < 128) : sa satInst satMem i k = b56 := by
  have := sIdx_lt hi hk
  simp only [sa, rdU8, Buf.ptr_addr]
  rw [satMem_read1 rfl (by simp [satInst, DType.bytes]; omega) (by simp [satInst, DType.bytes]; omega)]
  rfl

theorem sat_sb {j k : Nat} (hj : j < 128) (hk : k < 128) : sb satInst satMem j k = b56 := by
  have := sIdx_lt hj hk
  simp only [sb, rdU8, Buf.ptr_addr]
  rw [satMem_read1 rfl (by simp [satInst, DType.bytes]; omega) (by simp [satInst, DType.bytes]; omega)]
  rfl

theorem sat_scaled : scaled 1 b56 = 2 ^ (-71 : Int) := by
  rw [scaled, scaleV, b56_bits]; norm_num

theorem sat_dom1 : Rules.Num.e8m0Dom b56.bits.toNat ∧ Rules.Num.scaledDomBF16 (scaled 1 b56) := by
  refine ⟨by rw [b56_bits]; unfold Rules.Num.e8m0Dom; norm_num, Or.inr ?_⟩
  rw [sat_scaled]
  refine ⟨by rw [abs_of_pos (by positivity)]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num), ?_⟩
  rw [abs_of_pos (by positivity)]
  calc (2 : ℚ) ^ (-71 : Int) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)
    _ ≤ Math.Float.Format.bfloat16.maxVal := by decide +kernel

theorem sat_prods {i j : Nat} (hi : i < 128) (hj : j < 128) :
    ∀ k, k < 128 → prods satInst satMem i j k = 2 ^ (-71 : Int) * 2 ^ (-71 : Int) := by
  intro k hk
  simp only [prods, prodV, sat_xa hi hk, one8_val, sat_bVal hj hk, sat_sa hi hk, sat_sb hj hk,
    sat_scaled]

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨satInst, satMem, satInst_ok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid satMem := fun b h0 h1 h2 =>
    ⟨h0, h1, h2, fun _ _ => Option.some_ne_none _⟩
  refine ⟨hv _ (by norm_num [satInst]) (by norm_num [satInst, DType.bytes]) (by norm_num [satInst, DType.bytes]),
    hv _ (by norm_num [satInst_b_base]) (by rw [bDT_bytes, satInst_b_base, satInst_b_len]; norm_num)
      (by rw [bDT_bytes, satInst_b_base]; norm_num),
    hv _ (by norm_num [satInst]) (by norm_num [satInst, DType.bytes]) (by norm_num [satInst, DType.bytes]),
    hv _ (by norm_num [satInst]) (by norm_num [satInst, DType.bytes]) (by norm_num [satInst, DType.bytes]),
    hv _ (by norm_num [satInst]) (by norm_num [satInst, DType.bytes]) (by norm_num [satInst, DType.bytes]),
    ⟨fun i k hi hk => ?_, fun j k hj hk => ?_, fun i j hi hj => ?_⟩⟩
  · rw [sat_xa hi hk, sat_sa hi hk, one8_val]; exact ⟨one8_fin, sat_dom1⟩
  · rw [sat_bFin hj hk, sat_bVal hj hk, sat_sb hj hk]; exact ⟨rfl, sat_dom1⟩
  · have hp := sat_prods hi hj
    have hK : satInst.K = 128 := rfl
    have habs : dotAbs (prods satInst satMem i j) satInst.K = 128 * (2 ^ (-71 : Int) * 2 ^ (-71 : Int)) := by
      rw [dotAbs, hK, Finset.sum_congr rfl fun k hk => by rw [hp k (Finset.mem_range.mp hk)]]
      rw [Finset.sum_const, Finset.card_range, abs_of_pos (by positivity)]; simp
    have hch : chunkAbs (prods satInst satMem i j) 0 = 128 * (2 ^ (-71 : Int) * 2 ^ (-71 : Int)) := by
      rw [chunkAbs, Finset.sum_congr rfl fun k hk => by
        rw [hp (128 * 0 + k) (by have := Finset.mem_range.mp hk; omega)]]
      rw [Finset.sum_const, Finset.card_range, abs_of_pos (by positivity)]; simp
    have htol : tolOut satInst satMem i j = rho * (128 * (2 ^ (-71 : Int) * 2 ^ (-71 : Int))) + alpha := by
      rw [tolOut, hK, show 128 / 128 = 0 + 1 from rfl, tol, tol, partSum, hch]; simp
    rw [habs, htol, rho, alpha, Rules.Num.relDot16, Rules.Num.absDot16]
    norm_num

end Claims.BlockScaledMxfp8
