import TVA
import TVA.Lemmas.Claim
import TVA.Lemmas.Consistency
import TVA.Lemmas.Scalar
import TVA.Lemmas.DotLoop

/-!
# Grouped GEMM (tutorial 08): model, specification, conditions, assumptions

The source is `grouped_matmul_kernel` in `tutorials/08-grouped-gemm.py:93-169` (Triton 3.8.0),
launched by `group_gemm_fn` (`:172-213`) with `grid = (NUM_SM,)`:

```python
@triton.jit
def grouped_matmul_kernel(group_a_ptrs, group_b_ptrs, group_c_ptrs, group_gemm_sizes, g_lds,
                          group_size, NUM_SM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c)
            tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles
```

The host passes the matrix addresses as `int64` tensors (`torch.tensor(A_addrs)`), the sizes and
leading dimensions as `int32` tensors, and `group_size` as an `int32` scalar.
-/
namespace TVAExamples.GroupedGemm

open TVA

/-! ## Model (prover-written, line by line) -/

/-- `tl.cdiv(x, div)` of an `int32` scalar by a `constexpr`: `(x + (div - 1)) // div`
(`triton-3.8.0-standard.py:34-43`). Tensor `//` is `sdiv` (`semantic.py:317-326`). -/
def cdiv (x : I32) (div : Nat) : Prog I32 := do
  let q ← tl.idiv (Tensor.scalar (x + (IntV.ofNat (div - 1) : I32))) (Tensor.scalar (IntV.ofNat div : I32))
  pure q.item

/-- The `tl.dot` of the source: `float16` operands into a `float32` accumulator. The default
`input_precision` (`tf32`) only affects `float32` operands, and `max_num_imprecise_acc` is `0`
for non-fp8 operands (`semantic.py:1516-1521`). -/
abbrev cfg : DotCfg := { a := .f16, b := .f16, acc := .f32 }

/-- One iteration of the `K` loop, with the loop-carried accumulator and pointers. -/
def kstep (BM BN BK : Nat) (ldb : I32) (_kk : I32)
    (st : Tensor [BM, BN] F32 × Tensor [BM, BK] (Ptr .f16) × Tensor [BK, BN] (Ptr .f16)) :
    Prog (Tensor [BM, BN] F32 × Tensor [BM, BK] (Ptr .f16) × Tensor [BK, BN] (Ptr .f16)) := do
  let (accumulator, a_ptrs, b_ptrs) := st
  -- `tl.multiple_of(a_ptrs, [16, 16])`, `tl.multiple_of(b_ptrs, [16, 16])`: dropped. Both
  -- pointers are loop-carried (block arguments of the `scf.for` body), so Triton stores the
  -- hint as `tt.divisibility_arg<N>` on the loop (`triton-3.8.0-ir.cc:411-424`), which no pass
  -- reads (`triton-3.8.0-AxisInfo.cpp:1248-1253, 1289-1291, 1333-1338`).
  let a ← tl.load a_ptrs
  let b ← tl.load b_ptrs
  -- `accumulator += tl.dot(a, b)`: the combine pass may make it `tl.dot(a, b, accumulator)`
  let accumulator ← tl.dot_add cfg a b accumulator
  let a_ptrs := a_ptrs + tl.full [BM, BK] (IntV.ofNat BK : I32)
  let b_ptrs := b_ptrs + tl.full [BK, BN] ((IntV.ofNat BK : I32) * ldb)
  pure (accumulator, a_ptrs, b_ptrs)

/-- One tile: the `while` body. -/
def tile (g : I32) (gk num_n_tiles last_problem_end : I32) (group_a_ptrs group_b_ptrs group_c_ptrs : Ptr .i64)
    (g_lds : Ptr .i32) (NUM_SM BM BN BK : Nat) (tile_idx : I32) : Prog I32 := do
  let k := gk
  let lda ← tl.load1 (g_lds + g * 3)
  let ldb ← tl.load1 (g_lds + g * 3 + (1 : I32))
  let ldc ← tl.load1 (g_lds + g * 3 + (2 : I32))
  let a_ptr := tl.intToPtr .f16 (← tl.load1 (group_a_ptrs + g))
  let b_ptr := tl.intToPtr .f16 (← tl.load1 (group_b_ptrs + g))
  let c_ptr := tl.intToPtr .f16 (← tl.load1 (group_c_ptrs + g))
  let tile_idx_in_gemm := tile_idx - last_problem_end
  let tile_m_idx := (← tl.idiv (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar num_n_tiles)).item
  let tile_n_idx := (← tl.irem (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar num_n_tiles)).item
  let offs_am : Tensor [BM] I32 := tile_m_idx * (IntV.ofNat BM : I32) + tl.arange 0 BM
  let offs_bn : Tensor [BN] I32 := tile_n_idx * (IntV.ofNat BN : I32) + tl.arange 0 BN
  let offs_k : Tensor [BK] I32 := tl.arange 0 BK
  let a_ptrs : Tensor [BM, BK] (Ptr .f16) :=
    (a_ptr + offs_am.unsqueeze1 * lda).bcastCols BK + offs_k.unsqueeze0.bcastRows BM
  let b_ptrs : Tensor [BK, BN] (Ptr .f16) :=
    (b_ptr + offs_k.unsqueeze1 * ldb).bcastCols BN + offs_bn.unsqueeze0.bcastRows BK
  let accumulator := tl.zeros [BM, BN] .f32
  let kt ← cdiv k BK
  let (accumulator, _, _) ← tl.forRange (0 : I32) kt 1 (accumulator, a_ptrs, b_ptrs) (kstep BM BN BK ldb)
  let c ← tl.cvt (g := .binary16) accumulator
  let offs_cm : Tensor [BM] I32 := tile_m_idx * (IntV.ofNat BM : I32) + tl.arange 0 BM
  let offs_cn : Tensor [BN] I32 := tile_n_idx * (IntV.ofNat BN : I32) + tl.arange 0 BN
  let c_ptrs : Tensor [BM, BN] (Ptr .f16) :=
    (c_ptr + ldc * offs_cm.unsqueeze1).bcastCols BN + offs_cn.unsqueeze0.bcastRows BM
  tl.store c_ptrs c
  pure (tile_idx + (IntV.ofNat NUM_SM : I32))

/-- One problem of the group: the `for g` body. -/
def problem (group_a_ptrs group_b_ptrs group_c_ptrs : Ptr .i64) (group_gemm_sizes g_lds : Ptr .i32)
    (NUM_SM BM BN BK : Nat) (g : I32) (st : I32 × I32) : Prog (I32 × I32) := do
  let (tile_idx, last_problem_end) := st
  let gm ← tl.load1 (group_gemm_sizes + g * 3)
  let gn ← tl.load1 (group_gemm_sizes + g * 3 + (1 : I32))
  let gk ← tl.load1 (group_gemm_sizes + g * 3 + (2 : I32))
  let num_m_tiles ← cdiv gm BM
  let num_n_tiles ← cdiv gn BN
  let num_tiles := num_m_tiles * num_n_tiles
  -- `and` of two scalars: both sides are evaluated (`code_generator.py:1481-1515`)
  let tile_idx ← tl.whileLoop tile_idx
    (fun tile_idx => pure (last_problem_end.le tile_idx && tile_idx.lt (last_problem_end + num_tiles)))
    (tile g gk num_n_tiles last_problem_end group_a_ptrs group_b_ptrs group_c_ptrs g_lds NUM_SM BM BN BK)
  pure (tile_idx, last_problem_end + num_tiles)

def grouped_matmul_kernel (group_a_ptrs group_b_ptrs group_c_ptrs : Ptr .i64)
    (group_gemm_sizes g_lds : Ptr .i32) (group_size : I32)
    (NUM_SM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    let tile_idx := tl.program_id ctx 0
    let last_problem_end : I32 := 0
    let _ ← tl.forRange (0 : I32) group_size 1 (tile_idx, last_problem_end)
      (problem group_a_ptrs group_b_ptrs group_c_ptrs group_gemm_sizes g_lds NUM_SM
        BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K)
    pure ()

/-! ## Instances -/

/-- One GEMM of the group, `C = A·B`: `A` is `M × K`, `B` is `K × N`, `C` is `M × N`, all
`float16`, row-major with leading dimensions (row strides, in elements) `lda`, `ldb`, `ldc`.
Element `(i, k)` of `A` is element `i·lda + k` of the buffer `a`. -/
structure Gemm where
  M : Nat
  N : Nat
  K : Nat
  lda : Nat
  ldb : Nat
  ldc : Nat
  a : Buf .f16
  b : Buf .f16
  c : Buf .f16

/-- An instance: the group (problems `0 … G-1`), the configuration, and the five metadata
tensors (`group_a_ptrs`, …, `g_lds`). -/
structure Inst where
  G : Nat
  gemm : Nat → Gemm
  NUM_SM : Nat
  BM : Nat
  BN : Nat
  BK : Nat
  aptrs : Buf .i64
  bptrs : Buf .i64
  cptrs : Buf .i64
  sizes : Buf .i32
  lds : Buf .i32

/-- The number of tiles of problem `g`: `cdiv(M, BM) · cdiv(N, BN)`, i.e. `(M/BM)·(N/BN)` for
full tiles. -/
def Inst.tiles (d : Inst) (g : Nat) : Nat := (d.gemm g).M / d.BM * ((d.gemm g).N / d.BN)

/-- The first global tile index of problem `g` (`last_problem_end` when the loop reaches `g`). -/
def Inst.start (d : Inst) : Nat → Nat
  | 0 => 0
  | g + 1 => d.start g + d.tiles g

/-- The global tile holding element `(i, j)` of `C` of problem `g` (row-major tile order). -/
def Inst.tileOf (d : Inst) (g i j : Nat) : Nat :=
  d.start g + i / d.BM * ((d.gemm g).N / d.BN) + j / d.BN

/-! ## Numeric constants -/

/-- The assumed relative error of one `tl.dot` over `K` products (`Rules.Num.DotAcc`):
`(K + 2) · 2^-22`. -/
def rel (K : Nat) : ℚ := (K + 2) * 2 ^ (-22 : ℤ)

/-- The assumed absolute (underflow) error of one `tl.dot` over `K` products: `(K + 2) · 2^-126`. -/
def abs (K : Nat) : ℚ := (K + 2) * 2 ^ (-126 : ℤ)

/-- The relative error of one `accumulator += tl.dot(a, b)` step (`DotLoop.stepRel`). -/
def ρ (d : Inst) : ℚ := DotLoop.stepRel rel d.BK

/-- The absolute error of one step (`DotLoop.stepAbs`). -/
def α (d : Inst) : ℚ := DotLoop.stepAbs abs d.BK

/-- **The error bound of the `float32` result** of a problem with inner size `K`, for products
of absolute sum `S`: `((1 + ρ)^T − 1)·S + T·(1 + ρ)^T·α` with `T = K / BK` steps
(`DotLoop.bound`; at most `2T·(ρ·S + α)`, `errC_le`). -/
def errC (d : Inst) (K : Nat) (S : ℚ) : ℚ := DotLoop.bound (ρ d) (α d) (K / d.BK) S

/-! ## Conditions -/

/-- The conditions on one problem.
* Shapes: the buffers have `M·lda`, `K·ldb`, `M·ldc` elements, and the rows fit their leading
  dimension (`K ≤ lda`, `N ≤ ldb`, `N ≤ ldc`).
* **Full tiles**: the kernel has no masks ("assume full tile for now"), so `BM ∣ M`, `BN ∣ N`,
  `BK ∣ K`.
* `int32` arithmetic does not overflow: the sizes and leading dimensions are `int32` values, and
  so are `cdiv`'s sums and the offsets `row · lda`, `BK · ldb`, `ldc · row`.
* The accumulation stays in its accurate range: `(K / BK) · ρ ≤ 1/2`. -/
def Gemm.ok (d : Inst) (p : Gemm) : Prop :=
  p.a.len = p.M * p.lda ∧ p.b.len = p.K * p.ldb ∧ p.c.len = p.M * p.ldc ∧
  p.K ≤ p.lda ∧ p.N ≤ p.ldb ∧ p.N ≤ p.ldc ∧
  d.BM ∣ p.M ∧ d.BN ∣ p.N ∧ d.BK ∣ p.K ∧
  p.M + d.BM < 2 ^ 31 ∧ p.N + d.BN < 2 ^ 31 ∧ p.K + d.BK < 2 ^ 31 ∧
  p.lda < 2 ^ 31 ∧ p.ldb < 2 ^ 31 ∧ p.ldc < 2 ^ 31 ∧
  p.M * p.lda < 2 ^ 31 ∧ d.BK * p.ldb < 2 ^ 31 ∧ p.M * p.ldc < 2 ^ 31 ∧
  ((p.K / d.BK : Nat) : ℚ) * ρ d ≤ 1 / 2

/-- The instances covered.
* Every problem meets `Gemm.ok`.
* The block sizes are powers of two, with tiles of at most `2^20` elements (Triton's block
  limit); `NUM_SM ≥ 1`.
* The global tile indices and `g * 3 + 2` stay below `2^31` (`int32`): `3G < 2^31` and
  `start G + NUM_SM < 2^31`.
* The metadata tensors have their sizes (`G` addresses each, `3G` sizes, `3G` leading
  dimensions).
* **No aliasing of outputs**: each `C` is disjoint from every `A`, `B` and metadata tensor, and
  from the other problems' `C`. (Inputs may overlap one another.) -/
def Inst.ok (d : Inst) : Prop :=
  (∀ g, g < d.G → Gemm.ok d (d.gemm g)) ∧
  (∃ e, d.BM = 2 ^ e) ∧ (∃ e, d.BN = 2 ^ e) ∧ (∃ e, d.BK = 2 ^ e) ∧
  d.BM * d.BK ≤ 2 ^ 20 ∧ d.BK * d.BN ≤ 2 ^ 20 ∧ d.BM * d.BN ≤ 2 ^ 20 ∧
  0 < d.NUM_SM ∧ 3 * d.G < 2 ^ 31 ∧ d.start d.G + d.NUM_SM < 2 ^ 31 ∧
  d.aptrs.len = d.G ∧ d.bptrs.len = d.G ∧ d.cptrs.len = d.G ∧
  d.sizes.len = 3 * d.G ∧ d.lds.len = 3 * d.G ∧
  ∀ g, g < d.G → ∀ g', g' < d.G →
    Disjoint (d.gemm g).c.bytes (d.gemm g').a.bytes ∧ Disjoint (d.gemm g).c.bytes (d.gemm g').b.bytes ∧
    (g ≠ g' → Disjoint (d.gemm g).c.bytes (d.gemm g').c.bytes) ∧
    Disjoint (d.gemm g).c.bytes d.aptrs.bytes ∧ Disjoint (d.gemm g).c.bytes d.bptrs.bytes ∧
    Disjoint (d.gemm g).c.bytes d.cptrs.bytes ∧ Disjoint (d.gemm g).c.bytes d.sizes.bytes ∧
    Disjoint (d.gemm g).c.bytes d.lds.bytes

/-! ## Launch -/

/-- The launch: `grid = (NUM_SM,)`, and the source's block shapes: the offsets, the broadcast
offsets and pointers, the loaded tiles, the accumulator. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.NUM_SM
    [[d.BM], [d.BN], [d.BK], [d.BM, 1], [1, d.BK], [d.BK, 1], [1, d.BN],
      [d.BM, d.BK], [d.BK, d.BN], [d.BM, d.BN]]

/-- The kernel as launched on instance `d`. -/
def model (d : Inst) : Kernel :=
  grouped_matmul_kernel (d.aptrs.ptr 0) (d.bptrs.ptr 0) (d.cptrs.ptr 0) (d.sizes.ptr 0) (d.lds.ptr 0)
    (IntV.ofNat d.G) d.NUM_SM d.BM d.BN d.BK

/-! ## Specification -/

/-- The value of the `float16` element at `q` in `m` (`0` if it is not readable; `Pre` makes
every element the kernel reads readable and finite). -/
def val16 (m : Mem) (q : Ptr .f16) : ℚ :=
  match m.read .f16 q.addr with
  | some x => x.toRat
  | none => 0

/-- `A[i, k]` of problem `p` in memory `m`. -/
def Gemm.A (p : Gemm) (m : Mem) (i k : Nat) : ℚ := val16 m (p.a.ptr ((i * p.lda + k : ℕ) : ℤ))
/-- `B[k, j]`. -/
def Gemm.B (p : Gemm) (m : Mem) (k j : Nat) : ℚ := val16 m (p.b.ptr ((k * p.ldb + j : ℕ) : ℤ))
/-- The address of `C[i, j]`. -/
def Gemm.cAt (p : Gemm) (i j : Nat) : Int := (p.c.ptr ((i * p.ldc + j : ℕ) : ℤ)).addr

/-- The inputs: the metadata holds the problems' sizes, leading dimensions and addresses (as
the host builds them); `A`, `B`, `C` are valid buffers; the entries of `A` and `B` are finite. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  ∀ g, g < d.G → let p := d.gemm g
    init.read .i32 (d.sizes.ptr ((3 * g : ℕ) : ℤ)).addr = some (IntV.ofNat p.M) ∧
    init.read .i32 (d.sizes.ptr ((3 * g + 1 : ℕ) : ℤ)).addr = some (IntV.ofNat p.N) ∧
    init.read .i32 (d.sizes.ptr ((3 * g + 2 : ℕ) : ℤ)).addr = some (IntV.ofNat p.K) ∧
    init.read .i32 (d.lds.ptr ((3 * g : ℕ) : ℤ)).addr = some (IntV.ofNat p.lda) ∧
    init.read .i32 (d.lds.ptr ((3 * g + 1 : ℕ) : ℤ)).addr = some (IntV.ofNat p.ldb) ∧
    init.read .i32 (d.lds.ptr ((3 * g + 2 : ℕ) : ℤ)).addr = some (IntV.ofNat p.ldc) ∧
    (∃ x : IntV true 64, init.read .i64 (d.aptrs.ptr (g : ℤ)).addr = some x ∧ (x.bits.toNat : Int) = p.a.base) ∧
    (∃ x : IntV true 64, init.read .i64 (d.bptrs.ptr (g : ℤ)).addr = some x ∧ (x.bits.toNat : Int) = p.b.base) ∧
    (∃ x : IntV true 64, init.read .i64 (d.cptrs.ptr (g : ℤ)).addr = some x ∧ (x.bits.toNat : Int) = p.c.base) ∧
    p.a.Valid init ∧ p.b.Valid init ∧ p.c.Valid init ∧
    (∀ i k, i < p.M → k < p.K → ∃ x : F16, init.read .f16 (p.a.ptr ((i * p.lda + k : ℕ) : ℤ)).addr = some x ∧
      x.isFinite = true) ∧
    (∀ k j, k < p.K → j < p.N → ∃ x : F16, init.read .f16 (p.b.ptr ((k * p.ldb + j : ℕ) : ℤ)).addr = some x ∧
      x.isFinite = true)

/-- **The specification.** For every problem `g` and every element `(i, j)` of its `C`: the
`float16` result is the rounding (to nearest even, up to NaN payload) of a finite `float32`
value `o` within `errC K S` of the exact `Σₖ A[i,k]·B[k,j]`, where `S = Σₖ |A[i,k]·B[k,j]|`.
Nothing outside the `C` buffers changes. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ g, g < d.G → let p := d.gemm g
    ∀ i j, i < p.M → j < p.N → ∃ (o : F32) (c : F16),
      fin.read .f16 (p.cAt i j) = some c ∧ FloatV.NaNEq (FloatV.cvt o) c ∧ o.isFinite = true ∧
      |o.toRat - ∑ k ∈ Finset.range p.K, p.A init i k * p.B init k j| ≤
        errC d p.K (∑ k ∈ Finset.range p.K, |p.A init i k * p.B init k j|)) ∧
  ∀ x, (∀ g, g < d.G → x ∉ (d.gemm g).c.bytes) → fin x = init x

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## Assumptions (reviewed) -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on each covered instance the compiled kernel behaves as the
  model `model d` launched with `launch d`. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotTotal`: `tl.dot` of `float16` tiles into `float32` has a result. -/
  dotTotal : Rules.Num.DotTotal P cfg
  /-- [rule] `Num.DotAcc`: each element of `tl.dot(a, b, acc)` over `K` products is finite and
  within `(K + 2)·2^-22 · (|acc| + Σ|a·b|) + (K + 2)·2^-126` of `acc + Σ a·b`, when that fits
  `float32`. The `float16` products are exact in `float32`; the constant allows the tensor
  core's truncating alignment of every addend (`2^-23` each) twice over, plus two roundings. -/
  dotAcc : Rules.Num.DotAcc P cfg (fun _ => True) rel abs

end TVAExamples.GroupedGemm
