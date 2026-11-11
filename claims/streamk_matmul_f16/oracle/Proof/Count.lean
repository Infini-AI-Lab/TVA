import Claim
import TVA.Proof
import Proof.Addrs

/-!
# `Proof.Count`

The launch is valid; counting the runs of a tile, summing their errors over the programs, and reading memory on a span.
-/

/-!
# The launch of the model

The launch is a one-dimensional grid of `num_programs` programs whose block shapes are Triton's.
`launch_valid` is what `Compiler.Faithful` needs of it.
-/

namespace Claims.StreamKMatmul

open TVA

/-- The launch has `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- The nine block shapes of the launch are Triton's. -/
theorem blocks_ok :
    ∀ s ∈ ([[128], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 128], [64, 128],
      [128, 128]] : List Shape), s.TritonOk := by
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    subst hk
    exact ⟨7, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    subst hk
    exact ⟨6, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨7, by norm_num⟩
    · exact ⟨0, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨0, by norm_num⟩
    · exact ⟨6, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨7, by norm_num⟩
    · exact ⟨6, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨6, by norm_num⟩
    · exact ⟨0, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨0, by norm_num⟩
    · exact ⟨7, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨6, by norm_num⟩
    · exact ⟨7, by norm_num⟩
  · refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    rcases hk with rfl | rfl
    · exact ⟨7, by norm_num⟩
    · exact ⟨7, by norm_num⟩

/-- The launch is valid for a covered instance. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid :=
  Launch.valid_grid1 (num_programs_lt hd) blocks_ok

end Claims.StreamKMatmul


/-!
# Counting the programs' chunk runs, and reading memory

The chunk runs `[c0 p u, c1 p u)` of the programs on one tile `u` partition that tile's chunks
(`Index.chunk_partition`), so

* their lengths sum to `steps K` (the number of atomic additions one output element of a split
  tile receives, at most `steps K ≤ 128`), and
* summing `pre (c1) - pre (c0)` over the **nonempty** runs gives the whole dot sum, while summing
  the absolute values is bounded by `Σ_t |chunkSum t|`, and the `fineOne` errors add up to at most
  `tolDotR`.

A program whose chunk run is empty contributes nothing to any of these sums (both `c1 - c0` and
`pre (c1) - pre (c0)` vanish then).  Note that a program may run *past* the covered iterations, in
which case `c0 p u > c1 p u` is possible; its run is empty and it is excluded from the sums.

Also here: the sum form of `pre` (`pre_sub`), the monotonicity of `DotLoop.bound` in its magnitude
argument, and the congruence of a memory read with agreement on its span (the decomposition's `fin`
and the replayed memory agree on the output bytes).
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## Prefix sums -/

/-- `pre` is a running sum: the chunks of `[a, b)` add up to `pre b - pre a`. -/
theorem pre_sub {va vb : Nat → F16} {K a b : Nat} (h : a ≤ b) :
    pre va vb K b - pre va vb K a = ∑ t ∈ Finset.Ico a b, chunkSum va vb K t := by
  unfold pre
  rw [Finset.sum_Ico_eq_sub _ h]

/-- Triangle inequality: `|pre b - pre a|` is at most the sum of the chunk sums' magnitudes in between. -/
theorem abs_pre_sub_le {va vb : Nat → F16} {K a b : Nat} (h : a ≤ b) :
    |pre va vb K b - pre va vb K a| ≤ ∑ t ∈ Finset.Ico a b, |chunkSum va vb K t| := by
  rw [pre_sub h]
  exact Finset.abs_sum_le_sum_abs _ _

/-! ## The chunk runs of one tile -/

/-- A program that meets a tile has a nonempty range. -/
theorem lo_le_hi_of_c0_lt_c1 {d : Inst} {p u : Nat} (h : c0 d p u < c1 d p u) :
    lo d p ≤ hi d p := by
  unfold c0 c1 at h
  omega

/-- A program that meets a tile exists in the grid. -/
theorem lt_num_programs_of_c0_lt_c1 {d : Inst} (hd : d.ok) {p u : Nat}
    (h : c0 d p u < c1 d p u) : p < d.num_programs := by
  by_contra hp
  have hp' : d.num_programs ≤ p := Nat.le_of_not_lt hp
  have hle : hi d p ≤ lo d p := by
    have h1 : hi d p ≤ d.total_iters := hi_le_total d p
    have h2 : d.total_iters ≤ d.num_programs * d.iters_per_program := hd.cover
    have h3 : d.num_programs * d.iters_per_program ≤ p * d.iters_per_program :=
      Nat.mul_le_mul_right _ hp'
    unfold lo
    omega
  unfold c0 c1 at h
  omega

/-- The chunk runs of the programs on a tile are pairwise disjoint: a chunk has one program. -/
theorem c_range_disjoint (hd : d.ok) {u p q : Nat} (hu : u ∈ tiles d) (hne : p ≠ q) :
    Disjoint (Finset.Ico (c0 d p u) (c1 d p u)) (Finset.Ico (c0 d q u) (c1 d q u)) := by
  refine Finset.disjoint_left.mpr ?_
  intro v hp hq
  simp only [Finset.mem_Ico] at hp hq
  have hpT : c0 d p u < c1 d p u := lt_of_le_of_lt hp.1 hp.2
  have hqT : c0 d q u < c1 d q u := lt_of_le_of_lt hq.1 hq.2
  have hv : v < steps d.K := lt_of_lt_of_le hp.2 (c1_le (d := d) (p := p) (u := u))
  obtain ⟨r, ⟨hr, hc0, hc1⟩, huniq⟩ := chunk_partition hd hu hv
  have hpr : p = r := huniq p ⟨lt_num_programs_of_c0_lt_c1 hd hpT, hp.1, hp.2⟩
  have hqr : q = r := huniq q ⟨lt_num_programs_of_c0_lt_c1 hd hqT, hq.1, hq.2⟩
  exact hne (hpr.trans hqr.symm)

/-- The chunk runs of any set `S` of programs are pairwise disjoint. -/
theorem pairwiseDisjoint_c_range (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) (S : Finset Nat) :
    (↑S : Set Nat).PairwiseDisjoint (fun p => Finset.Ico (c0 d p u) (c1 d p u)) := by
  intro p _ q _ hpq
  exact c_range_disjoint hd hu hpq

/-- The chunk runs of a set `S` of programs that contains every program with a nonempty run cover
every chunk of the tile once. -/
theorem biUnion_c_range_of_subset (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) {S : Finset Nat}
    (hScov : ∀ p ∈ Finset.range d.num_programs, c0 d p u < c1 d p u → p ∈ S) :
    S.biUnion (fun p => Finset.Ico (c0 d p u) (c1 d p u)) = Finset.range (steps d.K) := by
  ext t
  simp only [Finset.mem_biUnion, Finset.mem_Ico, Finset.mem_range]
  constructor
  · rintro ⟨p, -, hc0, hc1⟩
    exact lt_of_lt_of_le hc1 (c1_le (d := d) (p := p) (u := u))
  · intro ht
    obtain ⟨p, ⟨hp, hc0, hc1⟩, -⟩ := chunk_partition hd hu ht
    exact ⟨p, hScov p (Finset.mem_range.mpr hp) (lt_of_le_of_lt hc0 hc1), hc0, hc1⟩

/-- The programs' chunk runs on tile `u` cover every chunk once. -/
theorem sum_c1_sub_c0 (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) :
    ∑ p ∈ Finset.range d.num_programs, (c1 d p u - c0 d p u) = steps d.K := by
  have hdisj := pairwiseDisjoint_c_range (d := d) hd hu (Finset.range d.num_programs)
  have hunion := biUnion_c_range_of_subset (d := d) hd hu
    (S := Finset.range d.num_programs) fun p hp _ => hp
  calc ∑ p ∈ Finset.range d.num_programs, (c1 d p u - c0 d p u)
      = ∑ p ∈ Finset.range d.num_programs, (Finset.Ico (c0 d p u) (c1 d p u)).card :=
          Finset.sum_congr rfl fun p _ => (Nat.card_Ico (c0 d p u) (c1 d p u)).symm
    _ = ((Finset.range d.num_programs).biUnion
          (fun p => Finset.Ico (c0 d p u) (c1 d p u))).card := (Finset.card_biUnion hdisj).symm
    _ = (Finset.range (steps d.K)).card := by rw [hunion]
    _ = steps d.K := Finset.card_range _

/-! ## Sums over the programs of one tile -/

/-- The number of programs that run a nonempty chunk range on tile `u` is at most `steps K`. -/
theorem sum_count_le_steps (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) :
    (∑ i : Fin (launch d).size, (if c0 d i.1 u < c1 d i.1 u then 1 else 0)) ≤ steps d.K := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i => if c0 d i u < c1 d i u then 1 else 0) (launch d).size, launch_size]
  calc ∑ p ∈ Finset.range d.num_programs, (if c0 d p u < c1 d p u then (1 : Nat) else 0)
      ≤ ∑ p ∈ Finset.range d.num_programs, (c1 d p u - c0 d p u) := by
        refine Finset.sum_le_sum fun p _ => ?_
        split_ifs with h
        · omega
        · omega
    _ = steps d.K := sum_c1_sub_c0 hd hu

/-- The nonempty runs of the programs on one tile sum to the whole dot sum. -/
theorem sum_pre_sub_split (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) {va vb : Nat → F16} :
    (∑ i : Fin (launch d).size,
      (if c0 d i.1 u < c1 d i.1 u then
        pre va vb d.K (c1 d i.1 u) - pre va vb d.K (c0 d i.1 u) else 0)) = dotSum va vb d.K := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i => if c0 d i u < c1 d i u then
      pre va vb d.K (c1 d i u) - pre va vb d.K (c0 d i u) else 0) (launch d).size,
    launch_size]
  set S : Finset Nat :=
    (Finset.range d.num_programs).filter (fun p => c0 d p u < c1 d p u) with hS
  have hfilter : (∑ p ∈ Finset.range d.num_programs,
        (if c0 d p u < c1 d p u then
          pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u) else 0))
      = ∑ p ∈ S, (pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u)) := by
    rw [hS, Finset.sum_filter]
  rw [hfilter]
  have hstep : ∀ p ∈ S, (pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u))
      = ∑ t ∈ Finset.Ico (c0 d p u) (c1 d p u), chunkSum va vb d.K t := by
    intro p hp
    rw [hS] at hp
    have hlt := (Finset.mem_filter.mp hp).2
    exact pre_sub (c0_le_c1 (lo_le_hi_of_c0_lt_c1 hlt))
  have hdisj := pairwiseDisjoint_c_range (d := d) hd hu S
  have hunion := biUnion_c_range_of_subset (d := d) hd hu (S := S) fun p hp hlt => by
    rw [hS]
    exact Finset.mem_filter.mpr ⟨hp, hlt⟩
  calc ∑ p ∈ S, (pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u))
      = ∑ p ∈ S, ∑ t ∈ Finset.Ico (c0 d p u) (c1 d p u), chunkSum va vb d.K t :=
          Finset.sum_congr rfl hstep
    _ = ∑ t ∈ S.biUnion (fun p => Finset.Ico (c0 d p u) (c1 d p u)),
          chunkSum va vb d.K t := (Finset.sum_biUnion hdisj).symm
    _ = ∑ t ∈ Finset.range (steps d.K), chunkSum va vb d.K t := by rw [hunion]
    _ = dotSum va vb d.K := pre_steps (va := va) (vb := vb) (K := d.K) hd.K_pos

/-- The absolute values of the programs' sums are bounded by the sum of the chunks' magnitudes. -/
theorem sum_abs_pre_sub_split (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) {va vb : Nat → F16} :
    (∑ i : Fin (launch d).size,
      (if c0 d i.1 u < c1 d i.1 u then
        |pre va vb d.K (c1 d i.1 u) - pre va vb d.K (c0 d i.1 u)| else 0))
      ≤ ∑ t ∈ Finset.range (steps d.K), |chunkSum va vb d.K t| := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i => if c0 d i u < c1 d i u then
      |pre va vb d.K (c1 d i u) - pre va vb d.K (c0 d i u)| else 0) (launch d).size,
    launch_size]
  set S : Finset Nat :=
    (Finset.range d.num_programs).filter (fun p => c0 d p u < c1 d p u) with hS
  have hfilter : (∑ p ∈ Finset.range d.num_programs,
        (if c0 d p u < c1 d p u then
          |pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u)| else 0))
      = ∑ p ∈ S, |pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u)| := by
    rw [hS, Finset.sum_filter]
  rw [hfilter]
  have hstep : ∀ p ∈ S, |pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u)|
      ≤ ∑ t ∈ Finset.Ico (c0 d p u) (c1 d p u), |chunkSum va vb d.K t| := by
    intro p hp
    rw [hS] at hp
    have hlt := (Finset.mem_filter.mp hp).2
    exact abs_pre_sub_le (c0_le_c1 (lo_le_hi_of_c0_lt_c1 hlt))
  have hdisj := pairwiseDisjoint_c_range (d := d) hd hu S
  have hunion := biUnion_c_range_of_subset (d := d) hd hu (S := S) fun p hp hlt => by
    rw [hS]
    exact Finset.mem_filter.mpr ⟨hp, hlt⟩
  calc ∑ p ∈ S, |pre va vb d.K (c1 d p u) - pre va vb d.K (c0 d p u)|
      ≤ ∑ p ∈ S, ∑ t ∈ Finset.Ico (c0 d p u) (c1 d p u), |chunkSum va vb d.K t| :=
          Finset.sum_le_sum hstep
    _ = ∑ t ∈ S.biUnion (fun p => Finset.Ico (c0 d p u) (c1 d p u)),
          |chunkSum va vb d.K t| := (Finset.sum_biUnion hdisj).symm
    _ = ∑ t ∈ Finset.range (steps d.K), |chunkSum va vb d.K t| := by rw [hunion]

/-- The `fineOne` errors of the programs' runs on one tile add up to at most `tolDotR`. -/
theorem sum_fineOne_tile (hd : d.ok) {u : Nat} (hu : u ∈ tiles d) {va vb : Nat → F16} :
    (∑ i : Fin (launch d).size, fineOne va vb d.K (c0 d i.1 u) (c1 d i.1 u - c0 d i.1 u))
      ≤ tolDotR va vb d.K := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i => fineOne va vb d.K (c0 d i u) (c1 d i u - c0 d i u)) (launch d).size,
    launch_size]
  set S : Finset Nat :=
    (Finset.range d.num_programs).filter (fun p => c0 d p u ≤ c1 d p u) with hS
  have hsub : S ⊆ Finset.range d.num_programs := by
    rw [hS]
    exact Finset.filter_subset _ _
  have hzero : ∀ p ∈ Finset.range d.num_programs, p ∉ S →
      fineOne va vb d.K (c0 d p u) (c1 d p u - c0 d p u) = 0 := by
    intro p hp hpn
    rw [hS] at hpn
    have hnle : ¬ (c0 d p u ≤ c1 d p u) := fun hle =>
      hpn (Finset.mem_filter.mpr ⟨hp, hle⟩)
    have hz : c1 d p u - c0 d p u = 0 := Nat.sub_eq_zero_of_le (Nat.le_of_not_le hnle)
    rw [hz]
    simp [fineOne]
  have hsub2 : (∑ p ∈ S, fineOne va vb d.K (c0 d p u) (c1 d p u - c0 d p u))
      = ∑ p ∈ Finset.range d.num_programs,
          fineOne va vb d.K (c0 d p u) (c1 d p u - c0 d p u) :=
    Finset.sum_subset hsub hzero
  rw [← hsub2]
  have hSle : ∀ p ∈ S, c0 d p u ≤ c1 d p u := by
    intro p hp
    rw [hS] at hp
    exact (Finset.mem_filter.mp hp).2
  have hTle : ∀ p ∈ S, c1 d p u ≤ steps d.K := fun p _ => c1_le
  have hpart : ∀ t, t < steps d.K → ∃! p, p ∈ S ∧ c0 d p u ≤ t ∧ t < c1 d p u := by
    intro t ht
    obtain ⟨p, ⟨hp, hc0, hc1⟩, huniq⟩ := chunk_partition hd hu ht
    refine ⟨p, ⟨?_, hc0, hc1⟩, ?_⟩
    · rw [hS]
      exact Finset.mem_filter.mpr
        ⟨Finset.mem_range.mpr hp, le_of_lt (lt_of_le_of_lt hc0 hc1)⟩
    · intro q hq
      rw [hS] at hq
      exact huniq q ⟨Finset.mem_range.mp (Finset.mem_filter.mp hq.1).1, hq.2.1, hq.2.2⟩
  exact sum_fineOne_le hSle hTle hpart

/-! ## The atomic bound is monotone in its magnitude argument -/

/-- `DotLoop.bound` is monotone in its magnitude argument. -/
theorem bound_mono_B {ρ α : ℚ} (hρ : 0 ≤ ρ) {t : Nat} {B B' : ℚ} (h : B ≤ B') :
    DotLoop.bound ρ α t B ≤ DotLoop.bound ρ α t B' := by
  unfold DotLoop.bound
  have hp : (1 : ℚ) ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have h0 : 0 ≤ (1 + ρ) ^ t - 1 := by linarith
  have := mul_le_mul_of_nonneg_left h h0
  linarith

/-! ## Reading memory on a span -/

/-- Reading bytes depends only on the bytes read. -/
theorem readBytes_congr {m m' : Mem} {a : Int} :
    ∀ n : Nat, (∀ i, i < n → m (a + i) = m' (a + i)) →
      m.readBytes a n = m'.readBytes a n
  | 0, _ => rfl
  | n + 1, h => by
    have h0 : m a = m' a := by simpa using h 0 (by omega)
    have h1 : m.readBytes (a + 1) n = m'.readBytes (a + 1) n :=
      readBytes_congr n fun i hi => by
        have := h (i + 1) (by omega)
        simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, add_left_comm] using this
    rw [Mem.readBytes, Mem.readBytes, h0, h1]

/-- Two memories that agree on an element's bytes read the same value there. -/
theorem read_congr {m m' : Mem} {dd : DType} {a : Int}
    (h : ∀ x ∈ Mem.span dd a, m x = m' x) : m.read dd a = m'.read dd a := by
  unfold Mem.read
  congr 2
  refine readBytes_congr dd.bytes fun i hi => h (a + i) ⟨by omega, by omega⟩

/-! ## Counting the logged additions -/

/-- A list of paired atomic steps has at most as many recorded `some` values at `q` as it has
steps at `q`. -/
theorem length_filterMap_le {l : List (AtomicStep × (Nat × Nat → Option ℚ))} (q : Nat × Nat) :
    (l.filterMap (fun p => p.2 q)).length ≤ (l.filter (fun p => (p.2 q).isSome)).length := by
  induction l with
  | nil => simp
  | cons p l ih =>
    cases hpq : p.2 q with
    | none =>
      simp [hpq]
      exact ih
    | some w =>
      simp [hpq]
      omega

end Claims.StreamKMatmul
