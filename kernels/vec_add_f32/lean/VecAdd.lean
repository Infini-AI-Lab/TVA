import TVA
import TVA.Lemmas.Block
import TVA.Lemmas.Claim
import TVA.Lemmas.Consistency

/-!
# Worked example: vector addition (tutorial 01)

A library test that runs the whole route for the simplest kernel: from the prover-written
model to the claim about the device. The source is `tutorials/01-vector-add.py`
(Triton 3.8.0):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

launched with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`.

The bundle states, in this order:

* **Model**: the kernel, line by line;
* **Conditions**: the instances covered (`Inst.ok`) and the inputs (`Pre`);
* **Specification**: `Post`, what memory holds afterwards (every output is the IEEE sum, nothing
  else changes), and `Spec`, the claim about the device;
* **Assumptions**: `Compiler.Faithful` only (IEEE addition needs no numeric rule);
* **Proof**: `main : Assumptions K P → Spec K`, and `consistent`.
-/
namespace TVAExamples.VecAdd

open TVA TVA.Block

/-! ## Model (prover-written, line by line) -/

def add_kernel (x_ptr y_ptr output_ptr : Ptr .f32) (n_elements : I32) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    let pid := tl.program_id ctx 0
    let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)
    let offsets := block_start + tl.arange 0 BLOCK_SIZE
    let mask := offsets.ltS n_elements
    let x ← tl.load (x_ptr + offsets) (some mask)
    let y ← tl.load (y_ptr + offsets) (some mask)
    let output ← tl.add x y          -- no product feeds this `+`: IEEE addition
    tl.store (output_ptr + offsets) output (some mask)

/-! ## Conditions -/

/-- An instance: the length, the block size, and the three buffers. -/
structure Inst where
  n : Nat
  BLOCK : Nat
  x : Buf .f32
  y : Buf .f32
  out : Buf .f32

/-- The instances covered: `n + BLOCK ≤ 2^31` (the `int32` offsets do not overflow), the buffers
have `n` elements, the output overlaps neither input (the inputs may overlap each other), and
`BLOCK_SIZE` is a power of two of at most `2^20` elements, as `tl.arange` and Triton's block
types require. -/
def Inst.ok (d : Inst) : Prop :=
  0 < d.BLOCK ∧ d.n + d.BLOCK ≤ 2 ^ 31 ∧ d.x.len = d.n ∧ d.y.len = d.n ∧ d.out.len = d.n ∧
    Disjoint d.x.bytes d.out.bytes ∧ Disjoint d.y.bytes d.out.bytes ∧
    (∃ e, d.BLOCK = 2 ^ e) ∧ d.BLOCK ≤ 2 ^ 20

/-! ## Launch -/

/-- `grid = (cdiv(n, BLOCK),)`. -/
def grid (d : Inst) : Nat := (d.n + d.BLOCK - 1) / d.BLOCK

/-- The launch: the grid, and the source's one block shape `[BLOCK_SIZE]` (`offsets`, `mask`,
`x`, `y`, `output`). -/
def launch (d : Inst) : Launch := Launch.grid1 (grid d) [[d.BLOCK]]

/-- The kernel as launched on instance `d`. -/
def model (d : Inst) : Kernel :=
  add_kernel (d.x.ptr 0) (d.y.ptr 0) (d.out.ptr 0) (IntV.ofNat d.n) d.BLOCK

/-! ## Conditions on the inputs -/

/-- The inputs: the three buffers are valid (aligned, present). Any values, NaN and infinities
included. -/
def Pre (d : Inst) (init : Mem) : Prop := d.x.Valid init ∧ d.y.Valid init ∧ d.out.Valid init

/-! ## Specification -/

/-- Each output element is the IEEE sum of the inputs (a NaN sum may be any NaN); nothing outside
`out` changes. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i : Nat, i < d.n → ∃ a b r, init.read .f32 (d.x.ptr i).addr = some a ∧
      init.read .f32 (d.y.ptr i).addr = some b ∧ fin.read .f32 (d.out.ptr i).addr = some r ∧
      FloatV.NaNEq (FloatV.add a b) r) ∧
    ∀ x, x ∉ d.out.bytes → fin x = init x

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## Assumptions (reviewed) -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on each covered instance the compiled kernel behaves as the
  model `model d` launched with `launch d`. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-! ## Proof -/

section Proof
variable (d : Inst)

/-- Program `p` reads its tiles of `x` and `y` … -/
def R (p : Nat) : Set Int := d.x.tile p d.BLOCK d.n ∪ d.y.tile p d.BLOCK d.n
/-- … and writes its tile of `out`. -/
def W (p : Nat) : Set Int := d.out.tile p d.BLOCK d.n

/-- What program `p` leaves in its own memory. -/
def Q (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ j, Tile p d.BLOCK d.n j → ∃ a b r, init.read .f32 (d.x.ptr j).addr = some a ∧
    init.read .f32 (d.y.ptr j).addr = some b ∧ m.read .f32 (d.out.ptr j).addr = some r ∧
    FloatV.NaNEq (FloatV.add a b) r

theorem size_launch : (launch d).size = grid d := by
  simp [launch, Launch.size, Launch.grid1]

/-- Program `i` runs the block program for `pid = i`. -/
theorem body_eq (i : Fin (launch d).size) : (model d).body ((launch d).ctx i) =
    (tl.load (d.x.ptr 0 + offs i d.BLOCK) (some (mask i d.BLOCK d.n)) >>= fun x =>
      tl.load (d.y.ptr 0 + offs i d.BLOCK) (some (mask i d.BLOCK d.n)) >>= fun y =>
      tl.add x y >>= fun output =>
      tl.store (d.out.ptr 0 + offs i d.BLOCK) output (some (mask i d.BLOCK d.n))) := by
  have hpid : ((launch d).ctx i).pid 0 = i :=
    Nat.mod_eq_of_lt (lt_of_lt_of_eq i.isLt (size_launch d))
  simp only [model, add_kernel, tl.program_id, hpid]
  rfl

/-- The launch is valid: the grid is below `2^31` and the block shape is one Triton accepts. -/
theorem launch_valid (hd : d.ok) : (launch d).Valid := by
  obtain ⟨hB, hnB, -, -, -, -, -, hpow, hle⟩ := hd
  have hg : grid d < 2 ^ 31 := by
    unfold grid
    calc (d.n + d.BLOCK - 1) / d.BLOCK ≤ d.n + d.BLOCK - 1 := Nat.div_le_self _ _
      _ < 2 ^ 31 := by omega
  refine ⟨hg, by simp [launch, Launch.grid1], by simp [launch, Launch.grid1], rfl, ?_⟩
  intro s hs
  simp only [launch, Launch.grid1, List.mem_singleton] at hs
  subst hs
  exact ⟨by simpa using hpow, by simpa using hle⟩

theorem tile_lt_grid {p j : Nat} (hd : d.ok) (hj : j < d.n) (h : Tile p d.BLOCK d.n j) : p < grid d := by
  have hB := hd.1
  rw [tile_unique h (tile_div hB hj), grid, Nat.div_lt_iff_lt_mul hB]
  have := Nat.lt_div_mul_add (a := d.n + d.BLOCK - 1) hB
  omega

end Proof

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent (model := model) (launch := launch) hA.faithful (fun _ _ => rfl)
    (fun d hd => launch_valid d hd) (fun d _ i => R d i) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_
  · -- each program, alone
    intro d hd init ⟨hvx, hvy, hvo⟩ i
    obtain ⟨hB, hnB, hx, hy, ho, -, -⟩ := hd
    have hp : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (size_launch d)
    have hfit := fits_of_grid hnB hp
    have hn : d.n < 2 ^ 31 := by omega
    rw [body_eq]
    refine safe_load_block d.x hfit hn hx.ge hvx Set.subset_union_left fun vx hvx' => ?_
    refine safe_load_block d.y hfit hn hy.ge hvy Set.subset_union_right fun vy hvy' => ?_
    refine safe_add fun out hout => safe_last ?_
    refine safe_store_block d.out hfit hn ho.ge hvo subset_rfl fun m' hm' _ _ => .ret ?_
    intro j hj
    obtain ⟨l, rfl⟩ := tile_lane hj
    exact ⟨_, _, _, hvx' l hj.2.2, hvy' l hj.2.2, hm' l hj.2.2, hout (l, ())⟩
  · -- footprints
    intro d hd init _ i j hij
    obtain ⟨-, -, hx, hy, ho, hxo, hyo, -⟩ := hd
    have hne : (i : Nat) ≠ j := Fin.val_ne_of_ne hij
    refine Set.disjoint_union_right.mpr ⟨Set.disjoint_union_right.mpr ⟨?_, ?_⟩, ?_⟩
    · exact Buf.tile_disjoint_of _ _ hxo.symm ho.ge hx.ge
    · exact Buf.tile_disjoint_of _ _ hyo.symm ho.ge hy.ge
    · exact d.out.tile_disjoint hne
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun j hj => ?_, fun x hx => hfr x fun i hi => hx (d.out.tile_sub_bytes hd.2.2.2.2.1.ge hi)⟩
    have htile := tile_div hd.1 hj
    have hp : j / d.BLOCK < (launch d).size := by
      rw [size_launch]; exact tile_lt_grid d hd hj htile
    obtain ⟨m, hQ, hag⟩ := hloc ⟨j / d.BLOCK, hp⟩
    obtain ⟨a, b, r, ha, hb, hm, hr⟩ := hQ j htile
    refine ⟨a, b, r, ha, hb, ?_, hr⟩
    rw [Mem.read_congr fun x hx => hag x (Or.inr (d.out.span_sub_tile htile hx))]
    exact hm

/-- **The assumptions are consistent**: the claim is not vacuous. -/
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

end TVAExamples.VecAdd
