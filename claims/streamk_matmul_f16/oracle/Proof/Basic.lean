import Claim
import TVA.Proof

/-!
# `Proof.Basic`

Iteration ranges of the programs, the chunks of the `K` loop, and the tolerance quantities (`mag16`, `pre`, `dev`) with their basic bounds.
-/

/-!
# The program's place in the grid, and the iteration ranges

`grid1` gives a 1-D grid, so program `i` of the launch has `pid i`.  Stream-K splits the
`total_iters = tilesM · tilesN · steps K` iterations of the `K` loop into contiguous ranges of
`iters_per_program` iterations, one per program: program `p` runs `[lo p, hi p)`.  This file
records the arithmetic of that partition.
-/

namespace Claims.StreamKMatmul

open TVA

/-! ## Basic size facts -/

/-- `total_iters` is at least one: the instance has `M`, `N`, `K` positive. -/
theorem total_pos {d : Inst} (hd : d.ok) : 0 < d.total_iters := by
  have hMp := hd.M_pos
  have hNp := hd.N_pos
  have hKp := hd.K_pos
  have hM : 0 < tilesM d := by unfold tilesM; omega
  have hN : 0 < tilesN d := by unfold tilesN; omega
  have hS : 0 < steps d.K := by unfold steps; omega
  rw [hd.total]
  exact Nat.mul_pos (Nat.mul_pos hM hN) hS

/-- `iters_per_program` is at least one, so a program's range is not empty. -/
theorem ipp_pos {d : Inst} (hd : d.ok) : 0 < d.iters_per_program := by
  have h := total_pos hd
  have hc := hd.cover
  exact Nat.pos_of_mul_pos_left (lt_of_lt_of_le h hc)

/-- The grid has fewer than `2^31` programs. -/
theorem num_programs_lt {d : Inst} (hd : d.ok) : d.num_programs < 2 ^ 31 := by
  have h := hd.progs_lt
  have hi := ipp_pos hd
  nlinarith

/-- `total_iters` is an `int32` value: at most `num_programs · iters_per_program`. -/
theorem total_lt {d : Inst} (hd : d.ok) : d.total_iters < 2 ^ 31 :=
  lt_of_le_of_lt hd.cover hd.progs_lt

/-! ## The grid -/

/-- The launch has `num_programs` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.num_programs := Launch.size_grid1 _ _ _

/-- Program `i` has `pid i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-! ## The iteration ranges -/

/-- The first iteration of program `p`. -/
def lo (d : Inst) (p : Nat) : Nat := p * d.iters_per_program

/-- One past the last iteration of program `p`. -/
def hi (d : Inst) (p : Nat) : Nat := min ((p + 1) * d.iters_per_program) d.total_iters

@[simp] theorem lo_zero (d : Inst) : lo d 0 = 0 := Nat.zero_mul _

/-- Consecutive programs' ranges start `iters_per_program` apart. -/
theorem lo_succ (d : Inst) (p : Nat) : lo d (p + 1) = lo d p + d.iters_per_program := by
  unfold lo; ring

/-- A program whose range starts inside `[0, total_iters)` has a non-empty range. -/
theorem lo_le_hi_of (d : Inst) {p : Nat} (h : lo d p ≤ d.total_iters) : lo d p ≤ hi d p := by
  refine le_min ?_ h
  unfold lo; rw [Nat.succ_mul]; exact Nat.le_add_right _ _

/-- A range ends within `total_iters`. -/
theorem hi_le_total (d : Inst) (p : Nat) : hi d p ≤ d.total_iters := min_le_right _ _

/-- A range ends within its nominal length. -/
theorem hi_le_succ (d : Inst) (p : Nat) : hi d p ≤ (p + 1) * d.iters_per_program := min_le_left _ _

/-- Range starts increase with the program index. -/
theorem lo_mono (d : Inst) {p q : Nat} (h : p ≤ q) : lo d p ≤ lo d q := by
  unfold lo; exact Nat.mul_le_mul_right _ h

/-- `(p+1) * iters_per_program = lo p + iters_per_program`. -/
theorem succ_mul_ipp (d : Inst) (p : Nat) : (p + 1) * d.iters_per_program = lo d p + d.iters_per_program := by
  unfold lo; ring

/-- The ranges tile `[0, total_iters)`: every iteration belongs to exactly one program. -/
theorem cover_all {d : Inst} (hd : d.ok) {it : Nat} (hit : it < d.total_iters) :
    ∃ p, p < d.num_programs ∧ lo d p ≤ it ∧ it < hi d p := by
  have hipp : 0 < d.iters_per_program := ipp_pos hd
  refine ⟨it / d.iters_per_program, ?_, Nat.div_mul_le_self _ _, ?_⟩
  · have h1 : it / d.iters_per_program * d.iters_per_program ≤ it := Nat.div_mul_le_self _ _
    have h2 : it < d.num_programs * d.iters_per_program := lt_of_lt_of_le hit hd.cover
    exact Nat.lt_of_mul_lt_mul_right (lt_of_le_of_lt h1 h2)
  · have hmod : it % d.iters_per_program < d.iters_per_program := Nat.mod_lt _ hipp
    have hdm : it / d.iters_per_program * d.iters_per_program + it % d.iters_per_program = it := by
      rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
    have hlt : it < (it / d.iters_per_program + 1) * d.iters_per_program := by
      rw [Nat.succ_mul]; omega
    exact lt_min hlt hit

/-- A program's range is a run of consecutive iterations. -/
theorem mem_range_iff {d : Inst} {p it : Nat} :
    (lo d p ≤ it ∧ it < hi d p) ↔
      (p * d.iters_per_program ≤ it ∧ it < (p + 1) * d.iters_per_program ∧
        it < d.total_iters) := by
  unfold lo hi
  omega

end Claims.StreamKMatmul


/-!
# The chunk decomposition of the `K` loop

`steps K = (K + 63) / 64` chunks of `64` consecutive `k`, the sums `chunkSum`, `chunkMag`, the
prefixes `pre`, the deviations `dev`, and the basic facts about them that the tolerance proof
uses.  Pure arithmetic: no memory, no kernel.
-/

namespace Claims.StreamKMatmul

open TVA

variable {va vb : Nat → F16} {K t : Nat}

/-! ## `steps` -/

/-- `steps K` is positive when `K` is. -/
theorem steps_pos {K : Nat} (hK : 0 < K) : 0 < steps K := by
  unfold steps
  omega

/-- `steps K ≤ 128` for `K ≤ 2^13`. -/
theorem steps_le_128 {K : Nat} (hK : K ≤ 2 ^ 13) : steps K ≤ 128 := by
  unfold steps
  omega

/-- A chunk lies in `range K`. -/
theorem chunk_subset_range (K t : Nat) : chunk K t ⊆ Finset.range K := by
  intro k hk
  simp only [chunk, Finset.mem_Ico, Finset.mem_range] at hk ⊢
  omega

/-- Distinct chunks are disjoint. -/
theorem chunk_disjoint {K s t : Nat} (h : s ≠ t) : Disjoint (chunk K s) (chunk K t) := by
  rcases Nat.lt_or_gt_of_ne h with hlt | hlt
  · refine Finset.disjoint_left.mpr fun k hk hk' => ?_
    simp only [chunk, Finset.mem_Ico] at hk hk'
    have : 64 * s + 64 ≤ 64 * t := by omega
    omega
  · refine Finset.disjoint_left.mpr fun k hk hk' => ?_
    simp only [chunk, Finset.mem_Ico] at hk hk'
    have : 64 * t + 64 ≤ 64 * s := by omega
    omega

/-- A product index in a chunk belongs to a chunk below `steps K`. -/
theorem mem_chunk_range {K t k : Nat} (h : k ∈ chunk K t) : t < steps K := by
  simp only [chunk, Finset.mem_Ico] at h
  unfold steps
  omega

/-- The chunks cover `range K`. -/
theorem biUnion_chunk {K : Nat} (hK : 0 < K) :
    Finset.range K = (Finset.range (steps K)).biUnion (fun t => chunk K t) := by
  ext k
  simp only [Finset.mem_range, Finset.mem_biUnion]
  constructor
  · intro hk
    refine ⟨k / 64, ?_, ?_⟩
    · have hK' : 0 < K := hK
      unfold steps
      omega
    · simp only [chunk, Finset.mem_Ico]; omega
  · rintro ⟨t, -, ht⟩
    exact Finset.mem_range.mp (chunk_subset_range K t ht)

/-- `pre (t+1) = pre t + chunkSum t`. -/
theorem pre_succ : pre va vb K (t + 1) = pre va vb K t + chunkSum va vb K t := by
  simp [pre, Finset.sum_range_succ]

/-- `pre` over all chunks is the whole dot product. -/
theorem pre_steps {K : Nat} (hK : 0 < K) : pre va vb K (steps K) = dotSum va vb K := by
  have hdisj : ∀ s ∈ Finset.range (steps K), ∀ u ∈ Finset.range (steps K),
      s ≠ u → Disjoint (chunk K s) (chunk K u) :=
    fun s _ u _ hsu => chunk_disjoint hsu
  unfold pre dotSum
  rw [biUnion_chunk hK]
  exact (Finset.sum_biUnion hdisj).symm

/-! ## `mag16` -/

/-- `mag16` is the library's `Device.liftMag .f16`. -/
theorem mag16_eq_liftMag (q : ℚ) : mag16 q = Device.liftMag .f16 q := by
  unfold mag16 Device.liftMag Device.liftFloor
  rfl

/-- `mag16` is nonnegative. -/
theorem mag16_nonneg (q : ℚ) : 0 ≤ mag16 q := by
  unfold mag16
  split
  · exact le_refl 0
  · exact le_trans (abs_nonneg q) (le_max_left _ _)

/-- `|q| ≤ mag16 q`. -/
theorem abs_le_mag16 (q : ℚ) : |q| ≤ mag16 q := by
  unfold mag16
  split
  · rename_i h
    simp [h]
  · exact le_max_left _ _

/-- `chunkMag` is nonnegative. -/
theorem chunkMag_nonneg (va vb : Nat → F16) (K t : Nat) : 0 ≤ chunkMag va vb K t :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (mag16_nonneg _) (mag16_nonneg _)

/-- `|chunkSum t| ≤ chunkMag t`. -/
theorem abs_chunkSum_le_chunkMag (va vb : Nat → F16) (K t : Nat) :
    |chunkSum va vb K t| ≤ chunkMag va vb K t := by
  unfold chunkSum chunkMag
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  exact (abs_mul _ _).trans_le (mul_le_mul (abs_le_mag16 _) (abs_le_mag16 _) (abs_nonneg _)
    (mag16_nonneg _))

/-! ## `pre` and `dev` -/

/-- `dev` is nonnegative. -/
theorem dev_nonneg (va vb : Nat → F16) (K t : Nat) : 0 ≤ dev va vb K t :=
  Finset.le_sup'_of_le (f := fun s => |pre va vb K t - pre va vb K s|) (b := 0) (by simp)
    (abs_nonneg _)

/-- `|pre t - pre s| ≤ dev t` for `s ≤ t`. -/
theorem le_dev {s t : Nat} (h : s ≤ t) : |pre va vb K t - pre va vb K s| ≤ dev va vb K t :=
  Finset.le_sup' (f := fun s => |pre va vb K t - pre va vb K s|) (by simp; omega)

/-- `|chunkSum (t-1)| ≤ dev t`. -/
theorem abs_chunkSum_le_dev {t : Nat} (ht : 1 ≤ t) :
    |chunkSum va vb K (t - 1)| ≤ dev va vb K t := by
  have h := le_dev (va := va) (vb := vb) (K := K) (s := t - 1) (t := t) (by omega)
  have h2 : pre va vb K t = pre va vb K (t - 1) + chunkSum va vb K (t - 1) := by
    simpa [Nat.sub_add_cancel ht] using pre_succ (va := va) (vb := vb) (K := K) (t := t - 1)
  simpa [h2, add_sub_cancel_left] using h

/-- `|pre s - pre t| ≤ dev t` for `s ≤ t`. -/
theorem abs_pre_sub_le_dev {s t : Nat} (h : s ≤ t) :
    |pre va vb K s - pre va vb K t| ≤ dev va vb K t := by
  simpa [abs_sub_comm] using le_dev (va := va) (vb := vb) (K := K) h

end Claims.StreamKMatmul

