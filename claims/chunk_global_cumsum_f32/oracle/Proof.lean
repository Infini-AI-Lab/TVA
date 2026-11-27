import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Trees
import Proof.Lane
import Proof.LaneGood
import Proof.TripLanes
import Proof.Trip
import Proof.TripWalk
import Proof.Loop
import Proof.Program

/-!
# The proof of `chunk_global_cumsum_f32`

Program `i` of the grid is the (sequence, head) pair `(i / H, i % H)`. It reads within
`R d` (all of `s`, and `cu_seqlens` on a variable-length launch) and writes the elements of its
sequence at its head, `W d i`; `Q d i` says each of them holds a good value. `correct_of_safe`
composes the programs:

1. `Basic`, `Geo`, `TripLanes`: the body unfolded, the launch, the layout and the `int64`
   arithmetic of the lanes;
2. `Program.safe_program`: the prologue, then `Loop.safe_loop`, whose trips
   (`TripWalk.safe_trip`) keep the invariant `Inv`: the running sum is a tree of IEEE additions
   over the blocks done (`Lane.ZTree`), and each stored lane joins its scan tree with it into a
   tree bounded against `tol` (`LaneGood.lane_good`, `Num.good_of_tree`, with
   `Device.SumDepth` for the `tl.sum`);
3. below: distinct programs write disjoint bytes of `o` (distinct heads, or sequences that do not
   overlap), nobody writes what anybody reads, and the programs' results give `Post`.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst}

/-- A sequence ends before a later one starts. -/
theorem hi_le_lo (hT : d.varlen = true → TablesOk d) {n n' : Nat} (hnn : n < n')
    (hn' : n' < nSeq d) : hi d n ≤ lo d n' := by
  cases hvv : d.varlen
  · rw [hi_false hvv, lo_false hvv]
    have : (n + 1) * d.T ≤ n' * d.T := Nat.mul_le_mul_right _ hnn
    have e : (n + 1) * d.T = n * d.T + d.T := by ring
    omega
  · rw [hi_true hvv, lo_true hvv]
    rw [nSeq_true hvv] at hn'
    exact Int.toNat_le_toNat (cuAt_mono_le (hT hvv) hnn (by omega))

/-- Program `n·H + h` handles sequence `n` and head `h`. -/
theorem seqOf_mk (hd : d.ok) {n h : Nat} (hh : h < d.H) : seqOf d (n * d.H + h) = n := by
  unfold seqOf
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hd.H_pos, Nat.div_eq_of_lt hh, Nat.zero_add]

theorem headOf_mk {n h : Nat} (hh : h < d.H) : headOf d (n * d.H + h) = h := by
  unfold headOf
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]

/-- A program's elements are elements of the written set. -/
theorem seqElems_sub (hd : d.ok) {i : Nat} (hi : i < gridSize d) : seqElems d i ⊆ written d := by
  rintro e ⟨p, h1, h2, rfl⟩
  exact ⟨seqOf d i, p, headOf d i, seqOf_lt hi, h1, h2, headOf_lt hd i, rfl⟩

/-- A program writes bytes of `o`. -/
theorem W_sub_o (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat} (hi : i < gridSize d) :
    W d i ⊆ d.o.bytes := by
  refine Buf.elems_sub_bytes d.o ?_
  rintro e ⟨p, h1, h2, rfl⟩
  have := elem_lt hd hT hi h2
  have := hd.o_len
  omega

/-- No program writes what a program reads. -/
theorem W_disj_R (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat} (hi : i < gridSize d) :
    Disjoint (W d i) (R d) := by
  rw [Set.disjoint_left]
  intro x hx hr
  have ho := W_sub_o hd hT hi hx
  unfold R at hr
  rcases hr with hr | hr
  · exact Set.disjoint_left.mp hd.o_s ho hr
  · split at hr
    · rename_i hv; exact Set.disjoint_left.mp (hd.o_cu hv) ho hr
    · exact hr

/-- Distinct programs write distinct elements. -/
theorem W_disj_W (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i j : Nat}
    (hi : i < gridSize d) (hj : j < gridSize d) (hij : i ≠ j) : Disjoint (W d i) (W d j) := by
  refine Buf.elems_disjoint d.o ?_
  rw [Set.disjoint_left]
  rintro e ⟨p, hp1, hp2, rfl⟩ ⟨p', hq1, hq2, he⟩
  have hH := hd.H_pos
  have hhi := headOf_lt hd i
  have hhj := headOf_lt hd j
  -- the heads and positions agree
  have hh : headOf d i = headOf d j := by
    have h1 : (p * d.H + headOf d i) % d.H = (p' * d.H + headOf d j) % d.H := by rw [he]
    rw [Nat.add_comm (p * d.H), Nat.add_mul_mod_self_right, Nat.add_comm (p' * d.H),
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hhi, Nat.mod_eq_of_lt hhj] at h1
    exact h1
  have hpp : p = p' := by
    rw [hh] at he
    have h1 : p' * d.H = p * d.H := by omega
    exact (Nat.eq_of_mul_eq_mul_right hH h1).symm
  subst hpp
  have hsi := seq_head (d := d) i
  have hsj := seq_head (d := d) j
  rcases Nat.lt_trichotomy (seqOf d i) (seqOf d j) with hlt | heq | hgt
  · have := hi_le_lo hT hlt (seqOf_lt hj)
    unfold loOf hiOf at *
    omega
  · apply hij
    rw [← hsi, ← hsj, heq, hh]
  · have := hi_le_lo hT hgt (seqOf_lt hi)
    unfold loOf hiOf at *
    omega

/-- **The programs' results give `Post`.** -/
theorem main_post (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m : Mem,
      Q d (i : Nat) m ∧ ∀ x ∈ R d ∪ W d (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d (i : Nat)) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro n p h hn hp1 hp2 hh
    have hig : n * d.H + h < gridSize d := by
      unfold gridSize
      have : (n + 1) * d.H ≤ nSeq d * d.H := Nat.mul_le_mul_right _ hn
      have e : (n + 1) * d.H = n * d.H + d.H := by ring
      omega
    have hfin : n * d.H + h < (launch d).size := by rw [launch_size]; exact hig
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨n * d.H + h, hfin⟩
    simp only [] at hQ hagree
    have hs := seqOf_mk hd (n := n) hh
    have hh' := headOf_mk (d := d) (n := n) hh
    have hlo : loOf d (n * d.H + h) = lo d n := by unfold loOf; rw [hs]
    have hhi : hiOf d (n * d.H + h) = hi d n := by unfold hiOf; rw [hs]
    obtain ⟨v, hread, hgood⟩ := hQ p (by rw [hlo]; exact hp1) (by rw [hhi]; exact hp2)
    rw [hh'] at hread
    rw [hs, hh'] at hgood
    refine ⟨v, ?_, hgood⟩
    rw [← hread]
    refine Mem.read_congr (fun x hx => hagree x (Or.inr ?_))
    refine Buf.span_sub_elems d.o (j := p * d.H + h) ?_ (by exact_mod_cast hx)
    exact ⟨p, by rw [hlo]; exact hp1, by rw [hhi]; exact hp2, by rw [hh']⟩
  · intro z hz
    refine hfr z (fun i hzi => hz ?_)
    have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
    exact d.o.elems_mono (seqElems_sub hd hi) hzi

/-- **The claim**: each program is safe (`safe_program`), footprints are disjoint, and
`main_post` turns the programs' results into `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hT : d.varlen = true → TablesOk d := fun hv => (hpre.2.2.2 hv).2.2
  refine correct_of_safe (fun _ => R d) (fun i => W d i) (fun i m => Q d i m)
    (launch_valid hd) ?_ ?_ ?_ ?_
  · intro i
    exact safe_program hd hpre hP (lt_of_lt_of_eq i.isLt (launch_size d)) (launch_pid d i)
  · intro i j hij
    have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < gridSize d := lt_of_lt_of_eq j.isLt (launch_size d)
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (W_disj_R hd hT hi) (W_disj_W hd hT hi hj hij')
  · intro i
    exact (W_disj_R hd hT (lt_of_lt_of_eq i.isLt (launch_size d))).symm
  · intro fin hloc hfr
    exact main_post hd hloc hfr

end Claims.ChunkGlobalCumsum
