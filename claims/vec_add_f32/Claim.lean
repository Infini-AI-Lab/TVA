import TVA

/-!
# `vec_add_f32`: vector addition, Triton tutorial 01

The kernel is `add_kernel` of `source/01-vector-add.py` (lines 29–54; it reads the module-level
names `triton` and `tl`, lines 23–24). The host `add` (lines 62–78) allocates
`output = torch.empty_like(x)` and launches a 1-D grid of `cdiv(n_elements, 1024)` programs with
`BLOCK_SIZE=1024`. The script calls it on `float32` vectors of `98432` elements (the test, lines
84–93) and of `2^12 … 2^27` elements (the benchmark, lines 107–135); `SETUP.md` lists them.

What is claimed: every covered launch ends normally, element `j < n` of `output` is the IEEE
`float32` sum `x[j] + y[j]` (round to nearest even, up to NaN payload), and every other byte is
unchanged.

The conditions: `BLOCK_SIZE` is a power of two at most `2^20`, `n < 2^31`, each buffer holds at
least `n` elements, and the first `n` elements of `output` share no byte with the first `n`
elements of `x` or of `y`. Every `float32` input is covered.

The only float operation is an IEEE `+`, so the specification uses no device fact.
-/

namespace Claims.VecAdd

open TVA

/-! ## The model -/

/-- `add_kernel` (source lines 29–54). The pointers are `*fp32`, `n_elements` is an `int32` (an
integer argument in `[2, 2^31)`, or `0`; at `1` it is a compile-time constant, which gives the same
model, see `model`), and `BLOCK_SIZE : tl.constexpr` is a Lean parameter. -/
def add_kernel (x_ptr y_ptr output_ptr : Ptr .f32) (n_elements : I32) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    let pid := tl.program_id ctx 0
    -- `pid * BLOCK_SIZE`: the constexpr `BLOCK_SIZE` is built in the dtype the product computes
    -- in, `int32` (the tensor `pid`'s), and the product is `int32` (wrapping).
    let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)
    let offsets := block_start + tl.arange 0 BLOCK_SIZE
    -- the scalar `n_elements` (`int32`) is compared with every lane, signed
    let mask := offsets.ltS n_elements
    -- masked loads without `other`: the disabled lanes are unspecified
    let x ← tl.load (x_ptr + offsets) (some mask)
    let y ← tl.load (y_ptr + offsets) (some mask)
    -- `x + y`: both operands are loaded values, no product feeds the addition, so the IEEE `tl.add`
    let output ← tl.add x y
    -- `output` is already `float32`, the pointer's dtype: no conversion
    tl.store (output_ptr + offsets) output (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `add_kernel`. -/
structure Inst where
  /-- `n_elements` (the host passes `output.numel()`). -/
  n : Nat
  /-- `BLOCK_SIZE` (the host passes `1024`). -/
  BLOCK_SIZE : Nat
  /-- `x_ptr`: a contiguous `float32` tensor. -/
  x : Buf .f32
  /-- `y_ptr`: a contiguous `float32` tensor. -/
  y : Buf .f32
  /-- `output_ptr`: a contiguous `float32` tensor. -/
  output : Buf .f32

/-- The instances covered: every block size Triton accepts, every `n` that `n_elements` holds as
an `int32` (`n < 2^31`), buffers of at least `n` elements, and an output whose first `n` elements
share no byte with the first `n` elements of either input. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK_SIZE` is a power of two (Triton requires it of `tl.arange`). -/
  block_pow2 : ∃ e, d.BLOCK_SIZE = 2 ^ e
  /-- `BLOCK_SIZE` is at most `2^20` elements (Triton's largest block). -/
  block_le : d.BLOCK_SIZE ≤ 2 ^ 20
  /-- `n < 2^31`: Triton passes `n_elements` as an `int32` (or, at `n = 1`, as a constant), and
  the `int32` offsets `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` of the grid `cdiv(n,
  BLOCK_SIZE)` do not wrap. -/
  n_lt : d.n < 2 ^ 31
  /-- `x` has at least `n` elements. -/
  x_len : d.n ≤ d.x.len
  /-- `y` has at least `n` elements. -/
  y_len : d.n ≤ d.y.len
  /-- `output` has at least `n` elements. -/
  output_len : d.n ≤ d.output.len
  /-- The first `n` elements of `output` share no byte with the first `n` elements of `x`. -/
  output_x : Disjoint (d.output.elems {j | j < d.n}) (d.x.elems {j | j < d.n})
  /-- The first `n` elements of `output` share no byte with the first `n` elements of `y`. -/
  output_y : Disjoint (d.output.elems {j | j < d.n}) (d.y.elems {j | j < d.n})

/-- The model on an instance: the kernel's arguments are the buffers' first elements, `n` as an
`int32`, and the block size. At `n = 1` Triton passes `n_elements` as the compile-time constant
`1`; its only use, `offsets < n_elements`, makes it an `int32` tensor before comparing
(`TVA.Triton.Promote`: a comparison first makes a Python number a tensor of its own dtype), so the
model is the same. -/
def model (d : Inst) : Kernel :=
  add_kernel (d.x.ptr 0) (d.y.ptr 0) (d.output.ptr 0) (IntV.ofNat d.n) d.BLOCK_SIZE

/-- The host's launch (source lines 70 and 75): a 1-D grid of `triton.cdiv(n_elements,
BLOCK_SIZE)` programs. The source's only block shape is `[BLOCK_SIZE]` (`tl.arange`, `offsets`,
`mask`, `x`, `y`, `output`). `enable_fp_fusion` is Triton's default (`true`); `grid1` keeps the
defaults `num_ctas = 1`, `launch_pdl = False` and `debug = false`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE) [[d.BLOCK_SIZE]] (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, and every byte
present) in the initial memory. Any `float32` values, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.y.Valid init ∧ d.output.Valid init

/-! ## The specification -/

/-- What the launch leaves: for each `j < n`, element `j` of `output` is the IEEE `float32` sum
(round to nearest even) of the initial elements `j` of `x` and `y`, up to NaN payload; every byte
outside `output`'s first `n` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ j : Nat, j < d.n → ∃ a b c : F32,
      init.read .f32 (d.x.ptr j).addr = some a ∧
      init.read .f32 (d.y.ptr j).addr = some b ∧
      fin.read .f32 (d.output.ptr j).addr = some c ∧
      FloatV.NaNEq (FloatV.add a b) c) ∧
    ∀ z, z ∉ d.output.elems {j | j < d.n} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `add_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The host's launch on `n` elements with `BLOCK_SIZE = 2^e`: `x`, `y` and a fresh `output` of `n`
elements each, at the 16-byte-aligned addresses `2^34`, `2·2^34` and `3·2^34`. -/
def mk (n e : Nat) : Inst where
  n := n
  BLOCK_SIZE := 2 ^ e
  x := ⟨2 ^ 34, n⟩
  y := ⟨2 * 2 ^ 34, n⟩
  output := ⟨3 * 2 ^ 34, n⟩

/-- Two buffers whose byte ranges do not meet have first `n` elements that do not meet. -/
theorem elems_disjoint_of_lt {b c : Buf .f32} {n : Nat} (hb : n ≤ b.len) (hc : n ≤ c.len)
    (h : b.base + b.len * 4 ≤ c.base ∨ c.base + c.len * 4 ≤ b.base) :
    Disjoint (b.elems {j | j < n}) (c.elems {j | j < n}) := by
  refine Buf.elems_disjoint_of b c ?_ (fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)
    (fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq, DType.bytes] at hx hy
  omega

/-- Every host launch on `n < 2^31` elements with a block size `2^e ≤ 2^20` is covered. -/
theorem mk_ok {n e : Nat} (hn : n < 2 ^ 31) (he : e ≤ 20) : (mk n e).ok where
  block_pow2 := ⟨e, rfl⟩
  block_le := Nat.pow_le_pow_right (by norm_num) he
  n_lt := hn
  x_len := le_refl _
  y_len := le_refl _
  output_len := le_refl _
  output_x := elems_disjoint_of_lt (le_refl _) (le_refl _) (Or.inr (by
    simp only [mk]; push_cast; omega))
  output_y := elems_disjoint_of_lt (le_refl _) (le_refl _) (Or.inr (by
    simp only [mk]; push_cast; omega))

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 98432 10, testMem, mk_ok (by norm_num) (by norm_num), ?_, ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [testMem]⟩

/-! ## Coverage of the script's launches (`SETUP.md`)

`add(x, y)` with `BLOCK_SIZE = 1024 = 2^10`: the test's `98432` elements (line 85) and the
benchmark's `2^12, …, 2^27` (line 110). -/

def setup_test_98432 : Inst := mk 98432 10
theorem setup_test_98432_ok : setup_test_98432.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p12 : Inst := mk (2 ^ 12) 10
theorem setup_bench_2p12_ok : setup_bench_2p12.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p13 : Inst := mk (2 ^ 13) 10
theorem setup_bench_2p13_ok : setup_bench_2p13.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p14 : Inst := mk (2 ^ 14) 10
theorem setup_bench_2p14_ok : setup_bench_2p14.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p15 : Inst := mk (2 ^ 15) 10
theorem setup_bench_2p15_ok : setup_bench_2p15.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p16 : Inst := mk (2 ^ 16) 10
theorem setup_bench_2p16_ok : setup_bench_2p16.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p17 : Inst := mk (2 ^ 17) 10
theorem setup_bench_2p17_ok : setup_bench_2p17.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p18 : Inst := mk (2 ^ 18) 10
theorem setup_bench_2p18_ok : setup_bench_2p18.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p19 : Inst := mk (2 ^ 19) 10
theorem setup_bench_2p19_ok : setup_bench_2p19.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p20 : Inst := mk (2 ^ 20) 10
theorem setup_bench_2p20_ok : setup_bench_2p20.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p21 : Inst := mk (2 ^ 21) 10
theorem setup_bench_2p21_ok : setup_bench_2p21.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p22 : Inst := mk (2 ^ 22) 10
theorem setup_bench_2p22_ok : setup_bench_2p22.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p23 : Inst := mk (2 ^ 23) 10
theorem setup_bench_2p23_ok : setup_bench_2p23.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p24 : Inst := mk (2 ^ 24) 10
theorem setup_bench_2p24_ok : setup_bench_2p24.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p25 : Inst := mk (2 ^ 25) 10
theorem setup_bench_2p25_ok : setup_bench_2p25.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p26 : Inst := mk (2 ^ 26) 10
theorem setup_bench_2p26_ok : setup_bench_2p26.ok := mk_ok (by norm_num) (by norm_num)
def setup_bench_2p27 : Inst := mk (2 ^ 27) 10
theorem setup_bench_2p27_ok : setup_bench_2p27.ok := mk_ok (by norm_num) (by norm_num)

end Claims.VecAdd
