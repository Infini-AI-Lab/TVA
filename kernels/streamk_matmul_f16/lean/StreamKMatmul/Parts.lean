import StreamKMatmul.Basic
import StreamKMatmul.Sched

/-!
# `streamk_matmul_f16`: the model in parts, and what the proof tracks

The proof splits the model's body into three programs (`iterProg`, one iteration of the while
loop of lines 41–64; `chunkProg` and `loadDot`, one trip of the inner loop of lines 51–57;
`tailProg` and `storeAdd`, lines 58–64), and `body_eq` shows, by `rfl`, that the model is their composition. These are proof
devices: the model is `streamk_matmul_kernel`.

What one program's proof tracks:
* `RowA`, `ColB`: a row of `a` and a column of `b` hold finite values in the initial memory;
* `AccInv`: the accumulator of the inner loop, within `DotLoop.bound` of its part's exact sum;
* `TileOut`: an element a program stores (a tile it owns);
* `LogInv`: the program's log of atomic steps holds, for each element of a split tile, one
  operand per part the program has done, within `DotLoop.bound` of the part's exact sum;
* `Q`: what a program leaves, and `LoopInv`: the invariant of the while loop.
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## The model in parts -/

/-- Lines 53–57 of a trip of the inner loop, with `offs_m`, `offs_n` of the iteration and
`offs_k` of the trip. -/
def loadDot (d : Inst) (offs_m offs_n : Tensor [128] I32) (offs_k : Tensor [64] I32) (acc : Tensor [128, 128] F32) :
    Prog (Tensor [128, 128] F32) := do
  let a ← tl.load (d.a.ptr 0 + offs_m.col 64 * (IntV.ofNat d.stride_am : I32) + offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))
    (some (((offs_m.col 64).ltS (IntV.ofNat d.M : I32)).and ((offs_k.row 128).ltS (IntV.ofNat d.K : I32))))
    (some (← tl.convert .f32 .f16 (tl.full [128, 64] (FloatV.litF32 (FloatV.ofPat 0)))))
  let b ← tl.load (d.b.ptr 0 + offs_k.col 128 * (IntV.ofNat d.stride_bk : I32) + offs_n.row 64 * (IntV.ofNat d.stride_bn : I32))
    (some (((offs_k.col 128).ltS (IntV.ofNat d.K : I32)).and ((offs_n.row 64).ltS (IntV.ofNat d.N : I32))))
    (some (← tl.convert .f32 .f16 (tl.full [64, 128] (FloatV.litF32 (FloatV.ofPat 0)))))
  let acc ← tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } a b acc
  return acc

/-- One trip of the inner loop (lines 52–57). -/
def chunkProg (d : Inst) (offs_m offs_n : Tensor [128] I32) (kk : I32) (acc : Tensor [128, 128] F32) :
    Prog (Tensor [128, 128] F32) :=
  loadDot d offs_m offs_n (kk * (IntV.ofNat 64 : I32) + tl.arange 0 64) acc

/-- Lines 60–64 of an iteration, with the pointers and the mask of lines 58–59. -/
def storeAdd (c_ptrs : Tensor [128, 128] (Ptr .f32)) (c_mask : Tensor [128, 128] Bool)
    (it k_first k_last iters_per_tile : I32) (acc : Tensor [128, 128] F32) : Prog I32 := do
  if IntV.eq k_first (IntV.ofInt 0 : I32) && IntV.eq k_last iters_per_tile then
    tl.store c_ptrs acc (some c_mask)
  else
    let _ ← tl.atomic_add c_ptrs acc (some c_mask)
    pure ()
  return it + (k_last - k_first)

/-- Lines 58–64 of an iteration. -/
def tailProg (d : Inst) (offs_m offs_n : Tensor [128] I32) (it k_first k_last iters_per_tile : I32)
    (acc : Tensor [128, 128] F32) : Prog I32 :=
  storeAdd (d.c.ptr 0 + offs_m.col 128 * (IntV.ofNat d.stride_cm : I32) + offs_n.row 128 * (IntV.ofNat d.stride_cn : I32))
    (((offs_m.col 128).ltS (IntV.ofNat d.M : I32)).and ((offs_n.row 128).ltS (IntV.ofNat d.N : I32)))
    it k_first k_last iters_per_tile acc

/-- One iteration of the while loop (lines 42–64). -/
def iterProg (d : Inst) (num_pid_n iters_per_tile «end» : I32) (it : I32) : Prog I32 := do
  let tile := (← tl.idiv (Tensor.scalar it) (Tensor.scalar iters_per_tile)).item
  let k_first := it - tile * iters_per_tile
  let k_last := (tl.iminimum (Tensor.scalar iters_per_tile) (Tensor.scalar (k_first + («end» - it)))).item
  let pid_m := (← tl.idiv (Tensor.scalar tile) (Tensor.scalar num_pid_n)).item
  let pid_n := (← tl.irem (Tensor.scalar tile) (Tensor.scalar num_pid_n)).item
  let offs_m : Tensor [128] I32 := pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128
  let offs_n : Tensor [128] I32 := pid_n * (IntV.ofNat 128 : I32) + tl.arange 0 128
  let acc ← tl.forRange k_first k_last (IntV.ofInt 1 : I32) (tl.zeros [128, 128] .f32) (chunkProg d offs_m offs_n)
  tailProg d offs_m offs_n it k_first k_last iters_per_tile acc

/-- The model's body is the composition of the parts (by `rfl`). -/
theorem body_eq (d : Inst) (ctx : Ctx) : (model d).body ctx = (do
    let pid := tl.program_id ctx 0
    let num_pid_n := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.N : I32) + (IntV.ofNat (128 - 1) : I32)))
      (Tensor.scalar (IntV.ofNat 128 : I32))).item
    let iters_per_tile := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.K : I32) + (IntV.ofNat (64 - 1) : I32)))
      (Tensor.scalar (IntV.ofNat 64 : I32))).item
    let it := pid * (IntV.ofNat d.iters_per_program : I32)
    let «end» := (tl.iminimum (Tensor.scalar (it + (IntV.ofNat d.iters_per_program : I32)))
      (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item
    let _ ← tl.whileLoop it (fun it => pure (IntV.lt it «end»)) (iterProg d num_pid_n iters_per_tile «end»)
    pure ()) := rfl

/-! ## What the proof tracks -/

/-- The constants of one fp16 `tl.dot` of `K = 64` (`Rules.Num.DotF16`). -/
abbrev rhoD : ℚ := Rules.Num.relDot16 64

abbrev alphaD : ℚ := Rules.Num.absDot16 64

/-- Row `r` of `a` holds finite values `va k` (`k < K`) in the initial memory. -/
def RowA (d : Inst) (init : Mem) (r : Nat) (va : Nat → F16) : Prop :=
  ∀ k, k < d.K → init.read .f16 (d.a.ptr ↑(r * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧
    (va k).isFinite = true

/-- Column `q` of `b` holds finite values `vb k` (`k < K`) in the initial memory. -/
def ColB (d : Inst) (init : Mem) (q : Nat) (vb : Nat → F16) : Prop :=
  ∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + q * d.stride_bn)).addr = some (vb k) ∧
    (vb k).isFinite = true

/-- The exact sum of the chunks `lo ≤ u < hi`. -/
def partS (d : Inst) (va vb : Nat → F16) (lo hi : Nat) : ℚ := ∑ u ∈ Finset.Ico lo hi, chunkS va vb d.K u

/-- The lifted magnitude of the chunks `lo ≤ u < hi`. -/
def partA (d : Inst) (va vb : Nat → F16) (lo hi : Nat) : ℚ := ∑ u ∈ Finset.Ico lo hi, chunkA va vb d.K u

/-- The accumulator after `cnt` chunks of tile `t`, from chunk `lo`: each enabled element whose row
of `a` and column of `b` are finite is finite and within `DotLoop.bound` of the chunks' exact sum. -/
def AccInv (d : Inst) (init : Mem) (t lo cnt : Nat) (acc : Tensor [128, 128] F32) : Prop :=
  ∀ x : Idx [128, 128], cEn d t x = true → ∀ va vb : Nat → F16,
    RowA d init (rowOf d t x.1) va → ColB d init (colOf d t x.2.1) vb →
      (acc.get x).isFinite = true ∧
        |(acc.get x).toRat - partS d va vb lo (lo + cnt)| ≤
          DotLoop.bound rhoD alphaD cnt (partA d va vb lo (lo + cnt))

/-- What an element of a stored tile holds in memory `m`: some value; and when its row of `a` and
column of `b` are finite, a finite value within `tolDot K A` of the exact sum. -/
def TileOut (d : Inst) (init m : Mem) (t : Nat) (x : Idx [128, 128]) : Prop :=
  ∃ c : F32, m.read .f32 (d.c.ptr ↑(cG d t x)).addr = some c ∧
    ∀ va vb : Nat → F16, RowA d init (rowOf d t x.1) va → ColB d init (colOf d t x.2.1) vb →
      c.isFinite = true ∧ |c.toRat - dotSum va vb d.K| ≤ tolDot d.K (dotMag va vb d.K)

/-- A logged step, paired with its operand, on the element `e` of `c`: a `float32` atomic add of
the operand (`AccStep` with `Rules.Num.AtomicAddF32`'s constants), or a step on other bytes. -/
abbrev AccRel (d : Inst) (e : Nat) : AtomicStep → Option ℚ → Prop :=
  AccStep .f32 (d.c.ptr (e : Int)).addr FloatV.toRat (fun z : F32 => z.isFinite = true)
    (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) Math.Float.Format.binary32.maxVal

/-- The program's tiles stored so far (those with `done t`) hold what `TileOut` says. -/
def MemInv (d : Inst) (init : Mem) (p : Nat) (done : Nat → Prop) (m : Mem) : Prop :=
  ∀ t x, owns d p t → done t → cEn d t x = true → TileOut d init m t x

/-- The program's log: for each element of a split tile whose row of `a` and column of `b` are
finite, the steps on it, paired with their operands: one operand `w` when the program has done its
part of the tile (`done t`), within `DotLoop.bound` of the part's exact sum, and none otherwise. -/
def LogInv (d : Inst) (init : Mem) (p : Nat) (done : Nat → Prop) (log : List AtomicStep) : Prop :=
  ∀ t x, ¬ whole d t → cEn d t x = true → ∀ va vb : Nat → F16,
    RowA d init (rowOf d t x.1) va → ColB d init (colOf d t x.2.1) vb →
    ∃ ops : Multiset ℚ, Paired (AccRel d (cG d t x)) log ops ∧
      (done t → ∃ w : ℚ, ops = {w} ∧
        |w - partS d va vb (kfirst d p t) (klast d p t)| ≤
          DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t))) ∧
      (¬ done t → ops = 0)

/-- What program `p` leaves: its tiles stored, its log of atomic steps on the split tiles. -/
def Q (d : Inst) (init : Mem) (p : Nat) (log : List AtomicStep) (m : Mem) : Prop :=
  MemInv d init p (touches d p) m ∧ LogInv d init p (touches d p) log

/-- The parts done once `it` is `n`. -/
def doneAt (d : Inst) (p n t : Nat) : Prop := touches d p t ∧ thi d p t ≤ n

/-- The variant of the while loop: `end − it`. -/
def mu (d : Inst) (p : Nat) (it : I32) : Nat := pend d p - it.toInt.toNat

/-- The invariant of the while loop: `it` is `n`, between the program's first iteration and its
end, at the first iteration or a tile's start; `a` and `b` unchanged and the buffers valid; the
parts done so far stored or logged. -/
def LoopInv (d : Inst) (init : Mem) (p : Nat) (it : I32) (log : List AtomicStep) (m : Mem) : Prop :=
  ∃ n : Nat, it.toInt = n ∧ pstart d p ≤ n ∧ (n ≤ pend d p ∨ n = pstart d p) ∧
    (n = pstart d p ∨ n % steps d.K = 0 ∨ pend d p ≤ n) ∧
    (∀ x ∈ R d, m x = init x) ∧ d.a.Valid m ∧ d.b.Valid m ∧ d.c.Valid m ∧
    MemInv d init p (doneAt d p n) m ∧ LogInv d init p (doneAt d p n) log

/-! ## Lanes -/

theorem arange_get (n : Nat) (i : Fin n) : ((tl.arange 0 n : Tensor [n - 0] I32).get (⟨i, by omega⟩, ())) =
    IntV.ofInt (((0 : Nat) : Int) + (i : Nat)) := rfl

/-- The lanes of `pid * B + tl.arange(0, B)`. -/
theorem lane_add {B : Nat} {pid : I32} {base : Nat} (hp : pid.toInt = base) (hB : base * B + B ≤ 2 ^ 31)
    (hB0 : B < 2 ^ 31) (i : Fin B) :
    (((pid * (IntV.ofNat B : I32) + tl.arange 0 B : Tensor [B - 0] I32) : Tensor [B] I32).get1 i).toInt =
      ((base * B + i : Nat) : Int) := by
  show (pid * IntV.ofNat B + IntV.ofInt (((0 : Nat) : Int) + (i : Nat))).toInt = _
  have h1 : (pid * (IntV.ofNat B : I32)).toInt = ((base * B : Nat) : Int) := by
    rw [i32_mul hp (i32_nat hB0)]
    · push_cast; ring
    · have : (0 : Int) ≤ base := Int.natCast_nonneg _
      have : (0 : Int) ≤ B := Int.natCast_nonneg _
      nlinarith
    · have : ((base * B : Nat) : Int) < 2 ^ 31 := by have := i.2; omega
      push_cast at this; linarith
  have h2 : (IntV.ofInt (((0 : Nat) : Int) + (i : Nat)) : I32).toInt = (i : Nat) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · simp
    · constructor <;> omega
  rw [i32_add h1 h2] <;> push_cast <;> omega

end Kernels.StreamKMatmul
