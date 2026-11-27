import Proof.Program

/-!
# `compute_position_i64`: `compute_position_kernel`

Program `b` serves request `b`: it reads its `extend_prefix_lens[b]` and the lengths `0 … b`, and
writes its positions `start(b) … start(b) + len(b) - 1` of `positions` and element `b` of
`extend_start_loc`.
The proof has three steps:

1. `Proof/Trip.lean`: one trip of the 512-wide store loop is safe (`safe_trip`);
2. `Proof/Program.lean`: the summing loop (`safe_sum`), the store loop and the start
   (`safe_tail`), and the whole body (`safe_program`);
3. `main`: by `correct_of_safe`; the positions of distinct requests do not meet because the
   running sums grow (`posElems_disjoint`), the outputs share no byte with the inputs, and the
   programs' results give `Post`.
-/

namespace Claims.ComputePosition

open TVA

/-! ## Footprints of distinct programs -/

/-- Distinct requests write distinct elements of `positions`. -/
theorem posElems_disjoint {d : Inst} {init : Mem} (hpre : Pre d init) {i j : Nat} (hij : i ≠ j)
    (hi : i < d.bs) (hj : j < d.bs) : Disjoint (posElems d init i) (posElems d init j) := by
  have key : ∀ {a c : Nat}, a < c → c < d.bs →
      Disjoint (posElems d init a) (posElems d init c) := by
    intro a c hac hc
    rw [Set.disjoint_left]
    rintro x ⟨p, hp, hx⟩ ⟨q, -, hx'⟩
    have h := start_succ_le hpre hac (le_of_lt hc)
    have hq : (0 : Int) ≤ (q : Int) := by exact_mod_cast Nat.zero_le q
    linarith
  rcases lt_or_gt_of_ne hij with h | h
  · exact key h hj
  · exact (key h hi).symm

/-- The bytes a program reads lie in the lengths, or in the prefixes when there are prefixes. -/
theorem R_sub {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.bs) {x : Int} (hx : x ∈ R d j) :
    x ∈ d.extend_seq_lens.bytes ∨ (d.hasPrefix = true ∧ x ∈ d.extend_prefix_lens.bytes) := by
  rcases hx with hx | hx
  · refine Or.inl (d.extend_seq_lens.elems_sub_bytes ?_ hx)
    intro k hk
    exact lt_of_le_of_lt hk (lt_of_lt_of_le hj hd.lens_len)
  · by_cases hP : d.hasPrefix = true
    · rw [ite_eq_left hP] at hx
      refine Or.inr ⟨hP, d.extend_prefix_lens.elems_sub_bytes ?_ hx⟩
      intro k hk
      rw [Set.mem_singleton_iff] at hk
      subst hk
      exact lt_of_lt_of_le hj (hd.prefix_len hP)
    · rw [ite_eq_right hP] at hx
      exact absurd hx (Set.notMem_empty x)

/-- The bytes a program writes lie in the two outputs. -/
theorem W_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat} (hi : i < d.bs)
    {x : Int} (hx : x ∈ W d init i) :
    x ∈ d.positions.bytes ∨ x ∈ d.extend_start_loc.bytes := by
  rcases hx with hx | hx
  · exact Or.inl (d.positions.elems_sub_bytes (posElems_len hpre hi) hx)
  · refine Or.inr (d.extend_start_loc.elems_sub_bytes ?_ hx)
    intro k hk
    rw [Set.mem_singleton_iff] at hk
    subst hk
    exact lt_of_lt_of_le hi hd.start_len

/-- No program writes what any program reads. -/
theorem W_disj_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i j : Nat}
    (hi : i < d.bs) (hj : j < d.bs) : Disjoint (W d init i) (R d j) := by
  obtain ⟨hPS, hPL, hSL, hPP⟩ := hd.disj
  rw [Set.disjoint_left]
  intro x hW hR
  rcases W_sub hd hpre hi hW with hx | hx <;> rcases R_sub hd hj hR with hy | ⟨hP, hy⟩
  · exact Set.disjoint_left.mp hPL hx hy
  · exact Set.disjoint_left.mp (hPP hP).1 hx hy
  · exact Set.disjoint_left.mp hSL hx hy
  · exact Set.disjoint_left.mp (hPP hP).2 hx hy

/-- Distinct programs write distinct bytes. -/
theorem W_disj_W {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i j : Nat}
    (hij : i ≠ j) (hi : i < d.bs) (hj : j < d.bs) : Disjoint (W d init i) (W d init j) := by
  have hsl : ∀ {k : Nat}, k < d.bs → ∀ k' ∈ ({k} : Set Nat), k' < d.extend_start_loc.len := by
    intro k hk k' hk'
    rw [Set.mem_singleton_iff] at hk'
    subst hk'
    exact lt_of_lt_of_le hk hd.start_len
  refine Disjoint.union_left (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_)
  · exact d.positions.elems_disjoint (posElems_disjoint hpre hij hi hj)
  · exact d.positions.elems_disjoint_of d.extend_start_loc hd.disj.1 (posElems_len hpre hi)
      (hsl hj)
  · exact (d.positions.elems_disjoint_of d.extend_start_loc hd.disj.1 (posElems_len hpre hj)
      (hsl hi)).symm
  · exact d.extend_start_loc.elems_disjoint (by simpa using hij)

/-! ## The claim -/

/-- A program index of the launch is a request of the batch. -/
theorem idx_lt {d : Inst} (i : Fin (launch d).size) : (i : Nat) < d.bs :=
  lt_of_lt_of_eq i.isLt (launch_size d)

/-- **The claim.** Each program is safe on its own footprints (`safe_program`), the footprints
are disjoint, and the programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d (i : Nat)) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    ?_ ?_ ?_
  · intro i j hij
    have hne : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (W_disj_R hd hpre (idx_lt i) (idx_lt j))
      (W_disj_W hd hpre hne (idx_lt i) (idx_lt j))
  · intro i
    exact (W_disj_R hd hpre (idx_lt i) (idx_lt i)).symm
  · intro fin hloc hfr
    refine ⟨?_, ?_, ?_⟩
    · intro b hb j hj
      obtain ⟨m, ⟨hQ, -⟩, hframe⟩ := hloc ⟨b, (launch_size d).symm ▸ hb⟩
      obtain ⟨v, hv, hvv⟩ := hQ j hj hj
      refine ⟨v, ?_, hvv⟩
      have hs0 := start_nonneg hpre (le_of_lt hb)
      have hjn : start d init b + (j : Int) = (((start d init b + (j : Int)).toNat : Nat) : Int) :=
        (Int.toNat_of_nonneg (by have := Nat.zero_le j; omega)).symm
      have hjin : (start d init b + (j : Int)).toNat ∈ posElems d init b :=
        ⟨j, hj, by rw [← hjn]⟩
      rw [Mem.read_congr (m₂ := m) fun x hx => hframe x (Or.inr (Or.inl ?_))]
      · exact hv
      · rw [hjn] at hx
        exact Buf.span_sub_elems d.positions hjin hx
    · intro b hb
      obtain ⟨m, ⟨-, v, hv, hvv⟩, hframe⟩ := hloc ⟨b, (launch_size d).symm ▸ hb⟩
      refine ⟨v, ?_, hvv⟩
      rw [Mem.read_congr (m₂ := m) fun x hx =>
        hframe x (Or.inr (Or.inr (Buf.span_sub_elems d.extend_start_loc (by simp) hx)))]
      exact hv
    · intro z hz1 hz2
      refine hfr z fun i hzi => ?_
      rcases hzi with ⟨j, hj, hx⟩ | ⟨j, hj, hx⟩
      · exact hz1 ⟨j, posElems_lt hpre (idx_lt i) hj, hx⟩
      · rw [Set.mem_singleton_iff] at hj
        subst hj
        exact hz2 ⟨(i : Nat), idx_lt i, hx⟩

end Claims.ComputePosition
