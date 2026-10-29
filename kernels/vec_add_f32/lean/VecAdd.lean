import TVA

/-!
# `vec_add_f32`: tutorial 01, `add_kernel` on `float32`

Source: `source/01-vector-add.py`. The kernel is lines 29–54, and it reads the module-level names
`triton` and `tl` (lines 23–24). Its host function `add` is lines 62–78, called on `float32`
vectors of `98432` elements (line 85) and of `2^12 … 2^27` elements (the benchmark, lines 110 and
121–122).

The claim: on every covered instance, every launch that runs ends normally, and the output's
first `n` elements are the IEEE `float32` sums `x[j] + y[j]` (up to NaN payload), with every
other byte the launch is given unchanged.

* The model (`add_kernel`, `model`, `launch`) is the source transcribed line by line (the model
  map is `model-map.json`).
* The conditions (`Inst`, `Inst.ok`, `Pre`) and the specification (`Post`, `Spec`).
* One assumption, `[rule] Compiler.Faithful`.
* The proof: `TVA.meets_independent`, one program at a time (`safe_program`); `consistent` and
  `satisfiable` (the source's own test instance).
-/

namespace Kernels.VecAdd

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
  BLOCK_SIZE)` do not wrap (`Block.fits_of_grid_pow2`). -/
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
`1`; its only use, `offsets < n_elements`, makes it an `int32` tensor before comparing (SEMANTICS.md,
"Constants"), so the model is the same. -/
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

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from which
  the model's launch is well behaved, is one of the model's launch. The rule covers a launch within
  its scope: part (A), the kernel and its launches, which the host's launches meet (README,
  Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-! ## The proof -/

section Proof

open Block

/-- The number of programs, `cdiv(n, BLOCK_SIZE)`. -/
abbrev grid (d : Inst) : Nat := (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE

/-- Program `p`'s body, with the block pattern of `TVA.Lemmas.Block`. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let x ← tl.load (d.x.ptr 0 + offs (c.pid 0) d.BLOCK_SIZE) (some (mask (c.pid 0) d.BLOCK_SIZE d.n))
        let y ← tl.load (d.y.ptr 0 + offs (c.pid 0) d.BLOCK_SIZE) (some (mask (c.pid 0) d.BLOCK_SIZE d.n))
        let output ← tl.add x y
        tl.store (d.output.ptr 0 + offs (c.pid 0) d.BLOCK_SIZE) output
          (some (mask (c.pid 0) d.BLOCK_SIZE d.n))) := rfl

theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_SIZE := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- The grid has fewer than `2^31` programs: `cdiv(n, BLOCK_SIZE) ≤ n` for `BLOCK_SIZE ≥ 1`. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := by
  have hB := block_pos hd
  have hn := hd.n_lt
  show (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE < 2 ^ 31
  rw [Nat.div_lt_iff_lt_mul hB]
  have : 2 ^ 31 * 1 ≤ 2 ^ 31 * d.BLOCK_SIZE := Nat.mul_le_mul_left _ hB
  omega

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk; exact hd.block_pow2
  · simpa using hd.block_le

/-- Every program of the grid has `int32` offsets that do not wrap. -/
theorem fits {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : Fits p d.BLOCK_SIZE := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have he20 : e ≤ 20 := by
    have h := hd.block_le
    rw [he] at h
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h
  exact fits_of_grid_pow2 he (by omega) hd.n_lt hp

/-- Every element below `n` is in the tile of a program of the grid. -/
theorem div_lt_grid {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.n) : j / d.BLOCK_SIZE < grid d := by
  have hB := block_pos hd
  show j / d.BLOCK_SIZE < (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.n + d.BLOCK_SIZE - 1) d.BLOCK_SIZE
  have h2 := Nat.mod_lt (d.n + d.BLOCK_SIZE - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- The read footprint of program `p`: its tiles of `x` and `y`. -/
def R (d : Inst) (p : Nat) : Set Int :=
  d.x.tile p d.BLOCK_SIZE d.n ∪ d.y.tile p d.BLOCK_SIZE d.n

/-- The write footprint of program `p`: its tile of `output`. -/
def W (d : Inst) (p : Nat) : Set Int := d.output.tile p d.BLOCK_SIZE d.n

/-- What program `p` leaves in its own memory `m`: each element of its tile of `output` is the
sum of the initial elements of `x` and `y`. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ j, Tile p d.BLOCK_SIZE d.n j → ∃ a b c : F32,
    init.read .f32 (d.x.ptr j).addr = some a ∧
    init.read .f32 (d.y.ptr j).addr = some b ∧
    m.read .f32 (d.output.ptr j).addr = some c ∧
    FloatV.NaNEq (FloatV.add a b) c

/-- **One program.** Program `p` of the grid, run alone from `init`, is safe, reads its tiles of
`x` and `y`, writes its tile of `output`, and leaves the sums there. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {p : Nat}
    (hp : p < grid d) :
    Safe E (R d p) (W d p) (fun _ m => Q d init p m) init
      (do
        let x ← tl.load (d.x.ptr 0 + offs p d.BLOCK_SIZE) (some (mask p d.BLOCK_SIZE d.n))
        let y ← tl.load (d.y.ptr 0 + offs p d.BLOCK_SIZE) (some (mask p d.BLOCK_SIZE d.n))
        let output ← tl.add x y
        tl.store (d.output.ptr 0 + offs p d.BLOCK_SIZE) output (some (mask p d.BLOCK_SIZE d.n))) := by
  obtain ⟨hvx, hvy, hvo⟩ := hpre
  have hfit : Fits p d.BLOCK_SIZE := fits hd hp
  have hn := hd.n_lt
  -- line 50: `x = tl.load(x_ptr + offsets, mask=mask)`
  refine safe_load_block d.x hfit hn hd.x_len hvx Set.subset_union_left fun vx hx => ?_
  -- line 51: `y = tl.load(y_ptr + offsets, mask=mask)`
  refine safe_load_block d.y hfit hn hd.y_len hvy Set.subset_union_right fun vy hy => ?_
  -- line 52: `output = x + y`
  refine safe_add fun r hr => ?_
  -- line 54: `tl.store(output_ptr + offsets, output, mask=mask)`
  refine safe_last (safe_store_block d.output hfit hn hd.output_len hvo subset_rfl
    fun m' hm' _ _ => safe_ret ?_)
  intro j hj
  obtain ⟨l, rfl⟩ := tile_lane hj
  have hl : p * d.BLOCK_SIZE + l < d.n := hj.2.2
  exact ⟨vx.get (l, ()), vy.get (l, ()), r.get (l, ()), hx l hl, hy l hl, hm' l hl, hr (l, ())⟩

/-- A tile of `b` and a tile of `c`, of any two programs, share no byte when the first `n`
elements of `b` and of `c` share none. -/
theorem tile_disjoint_in {b c : Buf .f32} {n B : Nat}
    (h : Disjoint (b.elems {j | j < n}) (c.elems {j | j < n})) (p q : Nat) :
    Disjoint (b.tile p B n) (c.tile q B n) :=
  h.mono (b.elems_mono fun _ hj => hj.2.2) (c.elems_mono fun _ hj => hj.2.2)

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ i => R d i) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    rw [body_eq, launch_pid]
    exact safe_program hd hpre _ (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    simp only [R, W]
    refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_
    · exact tile_disjoint_in hd.output_x _ _
    · exact tile_disjoint_in hd.output_y _ _
    · exact d.output.tile_disjoint hij'
  · -- no program reads what it writes
    intro d hd init _ i
    simp only [R, W]
    exact Disjoint.union_left (tile_disjoint_in hd.output_x _ _).symm
      (tile_disjoint_in hd.output_y _ _).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun j hj => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · have hB := block_pos hd
      have hp : j / d.BLOCK_SIZE < (launch d).size := by
        rw [launch_size]; exact div_lt_grid hd hj
      have ht : Tile (j / d.BLOCK_SIZE) d.BLOCK_SIZE d.n j := tile_div hB hj
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨j / d.BLOCK_SIZE, hp⟩
      obtain ⟨a, b, c, ha, hb, hc, hab⟩ := hQ j ht
      refine ⟨a, b, c, ha, hb, ?_, hab⟩
      rw [← hc]
      exact Mem.read_congr fun x hx => hagree x (Or.inr (d.output.span_sub_tile ht hx))
    · exact d.output.elems_mono (fun j hj => hj.2.2) hzi

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- The source's own test instance (lines 85–89): `n = 98432` and `BLOCK_SIZE = 1024`, with
`x`, `y` and `output` of `98432` elements each, at the 16-byte-aligned addresses `2^21`, `2·2^21`
and `3·2^21`. -/
def testInst : Inst where
  n := 98432
  BLOCK_SIZE := 1024
  x := ⟨2097152, 98432⟩
  y := ⟨4194304, 98432⟩
  output := ⟨6291456, 98432⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

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

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨testInst, testMem, ⟨⟨10, by norm_num [testInst]⟩, by norm_num [testInst],
    by norm_num [testInst], le_refl _, le_refl _, le_refl _, ?_, ?_⟩, ?_, ?_, ?_⟩
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by norm_num [testInst])
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by norm_num [testInst])
  all_goals
    refine ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes], by norm_num [testInst, DType.bytes],
      fun _ _ => by simp [testMem]⟩

end Proof

end Kernels.VecAdd
