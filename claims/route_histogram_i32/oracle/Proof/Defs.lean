import Claim
import TVA.Proof

/-!
# `route_histogram_i32`: definitions and the structural facts

The grid, the cells shared through the atomics, the per-program postcondition, and the launch
facts (`Valid`, `debug`, fitting block arithmetic).
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- Lane `l` of the mask `pid * BLOCK + arange < num_pairs`, as `Tensor.get` reads it. -/
theorem mask_get {p B n : Nat} (hfit : Fits p B) (hn : n < 2 ^ 31) (l : Fin (B - 0)) :
    (mask p B n).get (l, ()) = decide (p * B + l < n) := Block.mask_lane hfit hn l

/-- Lane `l` of `offs`, as `Tensor.get` reads it. -/
theorem lane_get {p B : Nat} (hfit : Fits p B) (l : Fin (B - 0)) :
    ((offs p B).get (l, ())).toInt = ((p * B + l : Nat) : Int) := Block.lane_toInt hfit l

/-- The number of programs, `cdiv(max(n, 1), BLOCK)`. -/
abbrev grid (d : Inst) : Nat := (max d.n 1 + d.block - 1) / d.block

/-- The cells shared through the atomics: the first `NUM_BUCKETS` elements of `counts`. -/
def cells (d : Inst) : Set (DType × Int) :=
  {c | ∃ b : Nat, b < d.numBuckets ∧ c = (DType.i32, (d.counts.ptr (b : Int)).addr)}

/-- The bytes of the cells are the bytes of the first `NUM_BUCKETS` elements of `counts`. -/
theorem cellBytes_cells (d : Inst) :
    cellBytes (cells d) = d.counts.elems {b : Nat | b < d.numBuckets} := by
  ext x
  constructor
  · rintro ⟨c, ⟨b, hb, rfl⟩, hx⟩
    exact ⟨b, hb, hx⟩
  · rintro ⟨b, hb, hx⟩
    exact ⟨_, ⟨b, hb, rfl⟩, hx⟩

/-- The bytes the private part of a program may read: the token slots, and the group ids when
the kernel has groups. -/
def Rpriv (d : Inst) : Set Int :=
  d.tokenSlots.bytes ∪ (if d.hasGroups = true then d.groupIds.bytes else ∅)

/-- The shared cells are bytes of `counts`. -/
theorem cellBytes_cells_le {d : Inst} (hd : d.ok) :
    cellBytes (cells d) ⊆ d.counts.bytes := by
  rw [cellBytes_cells]
  exact d.counts.elems_sub_bytes (fun j hj => Nat.lt_of_lt_of_le hj hd.counts_len)

/-- `counts` shares no byte with the inputs, so the shared cells are disjoint from what a program
reads privately. -/
theorem disjoint_cellBytes_Rpriv {d : Inst} (hd : d.ok) :
    Disjoint (cellBytes (cells d)) (Rpriv d) := by
  refine Set.disjoint_left.mpr ?_
  intro x hx1 hx2
  have h1 : x ∈ d.counts.bytes := cellBytes_cells_le hd hx1
  rcases hx2 with h2 | h2
  · exact Set.disjoint_left.mp hd.counts_ts h1 h2
  · by_cases hg : d.hasGroups = true
    · rw [ite_eq_left hg] at h2
      exact Set.disjoint_left.mp (hd.counts_gi hg) h1 h2
    · rw [ite_eq_right hg] at h2
      exact h2

/-- The increment an atomic step makes to the `int32` cell at `a`: `new - old` when the step is
on that cell, `0` otherwise. -/
def stepInc (a : Int) (s : AtomicStep) : I32 :=
  if h : s.d = DType.i32 ∧ s.a = a then
    (h.1 ▸ s.new : DType.i32.Val) - (h.1 ▸ s.old : DType.i32.Val)
  else 0

/-- A step on the cell itself increments it by `new - old`. -/
theorem stepInc_self {a : Int} (o n : I32) : stepInc a ⟨DType.i32, a, o, n⟩ = n - o := by
  simp [stepInc]

/-- A step on another cell increments this one by `0`. -/
theorem stepInc_ne {a a' : Int} (o n : I32) (ha : a' ≠ a) : stepInc a ⟨DType.i32, a', o, n⟩ = 0 := by
  simp [stepInc, ha]

/-- The number of program `p`'s lanes that are enabled and in bucket `b`. -/
def progCount (d : Inst) (p b : Nat) : Nat :=
  ((Finset.range d.block).filter
    (fun l => p * d.block + l < d.n ∧ bucketOf d (p * d.block + l) = b)).card

/-- What program `p` leaves in the shared cells: every step it logs is on a cell, and the log adds
its own pairs to each bucket. -/
def Qset (d : Inst) (p : Nat) (log : List AtomicStep) (_m : Mem) : Prop :=
  (∀ s ∈ log, (s.d, s.a) ∈ cells d) ∧
    ∀ b : Nat, b < d.numBuckets →
      (log.map (stepInc (d.counts.ptr (b : Int)).addr)).sum = IntV.ofNat (progCount d p b)

/-- `Qset` does not depend on the order of the log. -/
theorem Qset_perm (d : Inst) (p : Nat) :
    PermInv (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m) := by
  intro _ l l' m hpl h
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun s hs => h1 s (hpl.mem_iff.mpr hs), fun b hb => ?_⟩
  rw [← h2 b hb]
  exact (List.Perm.sum_eq (hpl.map (stepInc (d.counts.ptr (b : Int)).addr))).symm

/-! ## The launch -/

theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.block := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- `max n 1` fits `int32`. -/
theorem n_max_lt {d : Inst} (hd : d.ok) : max d.n 1 < 2 ^ 31 := by
  have := hd.n_lt; omega

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- Program `i` of the launch has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch has `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- The grid fits `int32`. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := by
  have hB := block_pos hd
  have hn := n_max_lt hd
  show (max d.n 1 + d.block - 1) / d.block < 2 ^ 31
  rw [Nat.div_lt_iff_lt_mul hB]
  have : 2 ^ 31 * 1 ≤ 2 ^ 31 * d.block := Nat.mul_le_mul_left _ hB
  omega

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (n := grid d)
    (blocks := if d.bins = 0 then [[d.block]] else [[d.block], [d.bins]]) (grid_lt hd) ?_
  by_cases hbin : d.bins = 0
  · simp only [hbin, ite_true]
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact ⟨fun k hk => by
      simp only [List.mem_singleton] at hk; subst hk; exact hd.block_pow2,
      by simpa using hd.block_le⟩
  · simp only [hbin, ite_false]
    intro s hs
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl
    · exact ⟨fun k hk => by
        simp only [List.mem_singleton] at hk; subst hk; exact hd.block_pow2,
        by simpa using hd.block_le⟩
    · obtain ⟨⟨e, he⟩, hble, -⟩ := hd.bins_ok.resolve_left hbin
      exact ⟨fun k hk => by
        simp only [List.mem_singleton] at hk; subst hk; exact ⟨e, he⟩,
        by simpa using hble⟩

/-- Every program of the grid has `int32` offsets that do not wrap. -/
theorem fits {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : Fits p d.block := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have he20 : e ≤ 20 := by
    have h := hd.block_le
    rw [he] at h
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h
  exact fits_of_grid_pow2 (n := max d.n 1) he (by omega) (n_max_lt hd) hp

/-- Every pair below `n` is in the block of a program of the grid. -/
theorem div_lt_grid {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.n) : j / d.block < grid d := by
  have hB := block_pos hd
  show j / d.block < (max d.n 1 + d.block - 1) / d.block
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (max d.n 1 + d.block - 1) d.block
  have h2 := Nat.mod_lt (max d.n 1 + d.block - 1) hB
  have : j < max d.n 1 := by omega
  rw [Nat.mul_comm] at h1
  omega

/-! ## The program -/

theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pair_ids := tl.program_id c 0 * (IntV.ofNat d.block : I32) + tl.arange 0 d.block
        let pair_mask := pair_ids.ltS (IntV.ofNat d.n)
        let bucket_ids ← route_bucket_ids (d.groupIds.ptr 0) (d.tokenSlots.ptr 0) pair_ids
          pair_mask d.gps d.maxLoras d.width d.hasGroups
        add_counts (d.counts.ptr 0) bucket_ids pair_mask d.numBuckets d.bins) := rfl

/-- The body of program `i`, with the lanes of the kernel written out. -/
theorem body_pid (d : Inst) (i : Fin (launch d).size) :
    (model d).body ((launch d).ctx i) =
      (do
        let pair_ids := offs (i : Nat) d.block
        let pair_mask := mask (i : Nat) d.block d.n
        let bucket_ids ← route_bucket_ids (d.groupIds.ptr 0) (d.tokenSlots.ptr 0) pair_ids
          pair_mask d.gps d.maxLoras d.width d.hasGroups
        add_counts (d.counts.ptr 0) bucket_ids pair_mask d.numBuckets d.bins) := by
  rw [body_eq]
  simp only [tl.program_id, launch_pid, offs, mask]

end Claims.RouteHistogram
