import Proof.Basic
import Proof.Pad
import Proof.Pos
import Proof.Req

/-!
# `prepare_pos_seq_lens_i32`: `_prepare_pos_seq_lens_kernel`

Program `r < N` serves request `r`: it reads its entries of `idx_mapping`, `query_start_loc` and
`num_computed_tokens`, and writes `seq_lens[r]` and its tokens of `pos`. Program `N` writes
`seq_lens[N .. M)`. The proof:

1. `Proof.Pad`: the last program (`safe_pad`);
2. `Proof.Pos`: the positions loop of a request (`safe_posProg`);
3. `Proof.Req`: a request's program (`safe_req`);
4. `main`: by `correct_of_safe`; the requests' windows of `pos` are disjoint because
   `query_start_loc` is nondecreasing, and their entries of `seq_lens` are below `N`, the last
   program's at or above.
-/

namespace Claims.PreparePosSeqLens

open TVA

/-! ## One program -/

/-- **One program** of the grid is safe from `init`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d init (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  have hiN : (i : Nat) < d.N + 1 := lt_of_lt_of_eq i.isLt (launch_size d)
  have hN := hd.grid_lt
  have hnr := toInt_numReqs hN
  rw [body_eq]
  simp only [tl.program_id, tl.num_programs, launch_pid, launch_grid]
  by_cases hi : (i : Nat) < d.N
  · have hne : (IntV.ofNat (i : Nat) : I32).eq (IntV.ofNat (d.N + 1) - (1 : I32)) = false := by
      rw [IntV.eq_eq, IntV.I32.toInt_ofNat (by omega), hnr]
      simp only [Nat.cast_inj, decide_eq_false_iff_not]
      omega
    rw [hne]
    simp only [Bool.false_eq_true, ↓reduceIte]
    exact safe_req hd hpre E i hi
  · have hiN' : (i : Nat) = d.N := by omega
    have heq : (IntV.ofNat (i : Nat) : I32).eq (IntV.ofNat (d.N + 1) - (1 : I32)) = true := by
      rw [IntV.eq_eq, IntV.I32.toInt_ofNat (by omega), hnr, hiN']
      simp
    rw [heq]
    simp only [↓reduceIte, R, W, Q, hi]
    exact safe_pad hd E init hpre.2.1 _ hnr

/-! ## The footprints -/

/-- The elements of `seq_lens` program `i` writes. -/
def SL (d : Inst) (i : Nat) : Set Nat := if i < d.N then {i} else padElems d

/-- The elements of `pos` program `i` writes. -/
def PL (d : Inst) (init : Mem) (i : Nat) : Set Nat := if i < d.N then posElems d init i else ∅

theorem W_eq (d : Inst) (init : Mem) (i : Nat) :
    W d init i = d.seq_lens.elems (SL d i) ∪ d.pos.elems (PL d init i) := by
  unfold W SL PL
  split_ifs
  · rfl
  · ext x
    simp [Buf.elems]

theorem SL_lt {d : Inst} (hd : d.ok) {i : Nat} :
    ∀ j ∈ SL d i, j < d.seq_lens.len := by
  intro j hj
  unfold SL at hj
  split_ifs at hj with h
  · rw [Set.mem_singleton_iff] at hj; have := hd.sl_N; omega
  · have := hd.sl_M; exact lt_of_lt_of_le hj.2 this

theorem PL_lt {d : Inst} {init : Mem} (hpre : Pre d init) {i : Nat} :
    ∀ j ∈ PL d init i, j < d.pos.len := by
  intro j hj
  unfold PL at hj
  split_ifs at hj with h
  · have h1 := qsl_le hpre (i + 1) d.N (by omega) (le_refl _)
    have h2 := hpre.2.2.2.2.2.2.2.1
    have h3 := hj.2
    have : (j : Int) < d.pos.len := by omega
    exact_mod_cast this
  · exact absurd hj (Set.notMem_empty j)

/-- What a program writes lies in the outputs. -/
theorem W_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (_hi : i < d.N + 1) : W d init i ⊆ d.seq_lens.bytes ∪ d.pos.bytes := by
  rw [W_eq]
  exact Set.union_subset_union (Buf.elems_sub_bytes _ (SL_lt hd (i := i)))
    (Buf.elems_sub_bytes _ (PL_lt hpre))

/-- What a program reads lies in the inputs. -/
theorem R_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Nat) :
    R d init i ⊆ d.idx_mapping.bytes ∪ d.query_start_loc.bytes ∪ d.num_computed_tokens.bytes := by
  unfold R
  split_ifs with h
  · refine Set.union_subset_union (Set.union_subset_union ?_ ?_) ?_
    · refine Buf.elems_sub_bytes _ fun j hj => ?_
      rw [Set.mem_singleton_iff] at hj; have := hd.idx_len; omega
    · refine Buf.elems_sub_bytes _ fun j hj => ?_
      have := hd.qsl_len
      rcases hj with rfl | hj
      · omega
      · rw [Set.mem_singleton_iff] at hj; omega
    · refine Buf.elems_sub_bytes _ fun j hj => ?_
      have := (hpre.2.2.2.2.2.2.2.2.2.1 i h).2
      have hj' : (j : Int) = slot d init i := hj
      omega
  · exact Set.empty_subset _

/-- The outputs share no byte with the inputs. -/
theorem out_in {d : Inst} (hd : d.ok) :
    Disjoint (d.seq_lens.bytes ∪ d.pos.bytes)
      (d.idx_mapping.bytes ∪ d.query_start_loc.bytes ∪ d.num_computed_tokens.bytes) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := hd.disj
  refine Disjoint.union_left ?_ ?_
  · exact Disjoint.union_right (Disjoint.union_right h5 h6) h7
  · exact Disjoint.union_right (Disjoint.union_right h2 h3) h4

theorem SL_disjoint {d : Inst} {i j : Nat} (hi : i < d.N + 1) (hj : j < d.N + 1) (hij : i ≠ j) :
    Disjoint (SL d i) (SL d j) := by
  unfold SL
  rw [Set.disjoint_left]
  intro k hk hk'
  split_ifs at hk hk' with h1 h2 h2
  · rw [Set.mem_singleton_iff] at hk hk'; omega
  · rw [Set.mem_singleton_iff] at hk; exact absurd hk'.1 (by omega)
  · rw [Set.mem_singleton_iff] at hk'; exact absurd hk.1 (by omega)
  · omega

theorem PL_disjoint {d : Inst} {init : Mem} (hpre : Pre d init) {i j : Nat} (hij : i ≠ j) :
    Disjoint (PL d init i) (PL d init j) := by
  unfold PL
  rw [Set.disjoint_left]
  intro k hk hk'
  split_ifs at hk hk' with h1 h2
  · obtain ⟨a1, a2⟩ := hk
    obtain ⟨b1, b2⟩ := hk'
    rcases Nat.lt_or_gt_of_ne hij with h | h
    · have := qsl_le hpre (i + 1) j (by omega) (by omega)
      omega
    · have := qsl_le hpre (j + 1) i (by omega) (by omega)
      omega
  · exact hk'
  · exact hk
  · exact hk

/-- Programs `i ≠ j` write disjoint bytes, and `i` writes nothing `j` reads. -/
theorem W_disj {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i j : Nat}
    (hi : i < d.N + 1) (hj : j < d.N + 1) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  refine Disjoint.union_right ?_ ?_
  · exact Disjoint.mono (W_sub hd hpre hi) (R_sub hd hpre j) (out_in hd)
  · rw [W_eq, W_eq]
    have hc := hd.disj.1
    refine Disjoint.union_left (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_)
    · exact Buf.elems_disjoint _ (SL_disjoint hi hj hij)
    · exact Disjoint.mono (Buf.elems_sub_bytes _ (SL_lt hd (i := i)))
        (Buf.elems_sub_bytes _ (PL_lt hpre)) hc.symm
    · exact Disjoint.mono (Buf.elems_sub_bytes _ (PL_lt hpre))
        (Buf.elems_sub_bytes _ (SL_lt hd (i := j))) hc
    · exact Buf.elems_disjoint _ (PL_disjoint hpre hij)

/-! ## The postcondition -/

theorem post_of {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Q d init (i : Nat) m ∧ ∀ x ∈ R d init (i : Nat) ∪ W d init (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init (i : Nat)) → fin x = init x) :
    Post d init fin := by
  have hsz := launch_size d
  refine ⟨fun r hr => ?_, fun r hr j hj => ?_, fun r hr1 hr2 => ?_, fun z hz => ?_⟩
  · obtain ⟨m, hQ, hag⟩ := hloc ⟨r, by omega⟩
    simp only [Q, hr, ↓reduceIte] at hQ
    obtain ⟨v, hv, hvv⟩ := hQ.1
    refine ⟨v, ?_, hvv⟩
    rw [← hv]
    refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
    simp only [W, hr, ↓reduceIte]
    exact Or.inl (Buf.span_sub_elems _ (Set.mem_singleton r) hx)
  · obtain ⟨m, hQ, hag⟩ := hloc ⟨r, by omega⟩
    simp only [Q, hr, ↓reduceIte] at hQ
    obtain ⟨v, hv, hvv⟩ := hQ.2 j hj
    refine ⟨v, ?_, hvv⟩
    rw [← hv]
    have h0 : 0 ≤ qsl d init r :=
      le_trans hpre.2.2.2.2.2.1 (qsl_le hpre 0 r (by omega) (by omega))
    obtain ⟨k, hk⟩ : ∃ k : Nat, (k : Int) = qsl d init r + (j : Int) :=
      ⟨_, Int.toNat_of_nonneg (by omega)⟩
    rw [← hk]
    refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
    simp only [W, hr, ↓reduceIte]
    refine Or.inr (Buf.span_sub_elems _ ?_ hx)
    show qsl d init r ≤ (k : Int) ∧ (k : Int) < qsl d init (r + 1)
    simp only [qlen] at hj
    omega
  · obtain ⟨m, hQ, hag⟩ := hloc ⟨d.N, by omega⟩
    simp only [Q, lt_irrefl, ↓reduceIte] at hQ
    obtain ⟨v, hv, hvv⟩ := hQ r hr1 hr2
    refine ⟨v, ?_, hvv⟩
    rw [← hv]
    refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
    simp only [W, lt_irrefl, ↓reduceIte]
    exact Buf.span_sub_elems _ (show d.N ≤ r ∧ r < d.M from ⟨hr1, hr2⟩) hx
  · refine hfr z fun i hzi => hz ?_
    have hi : (i : Nat) < d.N + 1 := lt_of_lt_of_eq i.isLt hsz
    rw [W_eq] at hzi
    rcases hzi with h | h
    · refine Or.inl (Buf.elems_mono _ ?_ h)
      intro k hk
      unfold SL at hk
      split_ifs at hk with h1
      · rw [Set.mem_singleton_iff] at hk; exact Or.inl (by omega)
      · exact Or.inr hk.2
    · refine Or.inr (Buf.elems_mono _ ?_ h)
      intro k hk
      unfold PL at hk
      split_ifs at hk with h1
      · exact ⟨i, h1, hk.1, hk.2⟩
      · exact absurd hk (Set.notMem_empty k)

/-! ## The claim -/

/-- **The claim.** Each program is safe on its own footprints (`safe_program`), the footprints
are disjoint, and the programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hsz := launch_size d
  refine correct_of_safe (fun i => R d init (i : Nat)) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    (fun i j hij => W_disj hd hpre (by omega) (by omega) (fun h => hij (Fin.ext h)))
    (fun i => ?_) (fun fin hloc hfr => post_of hd hpre hloc hfr)
  have hi : (i : Nat) < d.N + 1 := lt_of_lt_of_eq i.isLt hsz
  exact Disjoint.symm (Disjoint.mono (W_sub hd hpre hi) (R_sub hd hpre _) (out_in hd))

end Claims.PreparePosSeqLens
