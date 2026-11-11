import Proof.Lane

/-!
# `route_histogram_i32`: from the shared-cell facts to `Post`

`correct_of_safeA` hands back, per program, that the final memory has the program's `Qset` on its
reads and the replay of the whole log on the shared cells, and that the rest is unchanged. These
lemmas turn that into `Post`: each bucket's element is its initial value plus its pairs, and
every byte of `counts` past the buckets is unchanged.
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- Two distinct cells of `counts` have disjoint spans. -/
theorem cells_distinct (d : Inst) :
    ∀ c ∈ cells d, ∀ c' ∈ cells d,
      (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c' := by
  intro c hc c' hc' hne
  obtain ⟨b, hb, rfl⟩ := hc
  obtain ⟨b', hb', rfl⟩ := hc'
  obtain ⟨y, hy⟩ := hne
  rw [Set.mem_inter_iff] at hy
  by_cases hbb : b = b'
  · rw [hbb]
  · exfalso
    have hij : (b : Int) ≠ (b' : Int) := by
      intro h
      exact hbb (by omega)
    have hdis := Buf.span_disjoint d.counts hij
    exact Set.disjoint_left.mp hdis hy.1 hy.2

/-- The shared cells are present initially (`counts` is valid and holds the buckets). -/
theorem cells_present {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) :
    ∀ x ∈ cellBytes (cells d), init x ≠ none := by
  intro x hx
  rw [cellBytes_cells] at hx
  have hsub : d.counts.elems {b : Nat | b < d.numBuckets} ⊆ d.counts.bytes :=
    d.counts.elems_sub_bytes (fun j hj => Nat.lt_of_lt_of_le hj hd.counts_len)
  obtain ⟨-, hv, -⟩ := hpre
  exact hv.2.2.2 x (hsub hx)

/-- Every step of the group is in the log of some program. -/
theorem mem_proj_of_mem_map {N : Nat} (G : List (Fin N × AtomicStep)) {s : AtomicStep}
    (h : s ∈ G.map Prod.snd) : ∃ i : Fin N, s ∈ proj G i := by
  induction G with
  | nil => simp at h
  | cons g G ih =>
    obtain ⟨j, k⟩ := g
    simp only [List.map_cons, List.mem_cons] at h
    rcases h with hk | h
    · refine ⟨j, ?_⟩
      have hp : proj ((j, k) :: G) j = k :: proj G j := by simp [proj]
      rw [hp, hk]
      exact List.mem_cons_self ..
    · obtain ⟨i, hi⟩ := ih h
      refine ⟨i, ?_⟩
      by_cases hji : j = i
      · have hp : proj ((j, k) :: G) i = k :: proj G i := by simp [proj, hji]
        rw [hp]
        exact List.mem_cons_of_mem _ hi
      · have hp : proj ((j, k) :: G) i = proj G i := by simp [proj, hji]
        rw [hp]
        exact hi

/-- `int32` arithmetic: adding back what was taken away. -/
private theorem intV_add_sub_self {s : Bool} {w : Nat} (a b : IntV s w) : a + (b - a) = b := by
  apply IntV.ext
  show a.bits + (b.bits - a.bits) = b.bits
  rw [BitVec.add_comm]
  exact BitVec.sub_add_cancel _ _

/-- A step reaching the cell `(i32, a)` adds `stepInc a` to it; a step elsewhere leaves it. -/
private theorem mapsCell_stepInc {d : Inst} {a : Int}
    (hx : (DType.i32, a) ∈ cells d) {s : AtomicStep}
    (hs : (s.d, s.a) ∈ cells d) :
    MapsCell DType.i32 a (· + stepInc a s) s := by
  obtain ⟨d0, a0, old, new⟩ := s
  obtain ⟨k, hk, hEq⟩ := hs
  have hd0 : d0 = DType.i32 := congrArg Prod.fst hEq
  subst hd0
  have hs' : (DType.i32, a0) ∈ cells d := ⟨k, hk, hEq⟩
  by_cases ha : a0 = a
  · rw [ha]
    refine mapsCell_self ?_
    rw [stepInc_self]
    exact intV_add_sub_self old new
  · have hzero : stepInc a ⟨DType.i32, a0, old, new⟩ = 0 := stepInc_ne old new ha
    have hdis : Disjoint (Mem.span DType.i32 a0) (Mem.span DType.i32 a) := by
      refine Set.disjoint_left.mpr ?_
      intro y hy1 hy2
      exact ha (congrArg Prod.snd (cells_distinct d (DType.i32, a0) hs' (DType.i32, a) hx ⟨y, hy1, hy2⟩))
    rw [hzero]
    exact mapsCell_other_add hdis

/-- The shared-cell facts of `TVA.shared` imply `Post`. -/
theorem post_of_shared {d : Inst} (hd : d.ok) {init fin : Mem}
    {G : List (Fin (launch d).size × AtomicStep)} {mA : Mem}
    (hQ : ∀ i : Fin (launch d).size, ∃ m : Mem, Qset d (i : Nat) (proj G i) m ∧
      ∀ x ∈ Rpriv d, fin x = m x)
    (hrep : Replay init (G.map Prod.snd) mA)
    (hAgree : ∀ x ∈ cellBytes (cells d), fin x = mA x)
    (hframe : ∀ x, x ∉ cellBytes (cells d) → fin x = init x)
    (hinit : ∀ b, b < d.numBuckets →
      init.read DType.i32 (d.counts.ptr (b : Int)).addr = some (d.cntVal b)) :
    Post d init fin := by
  constructor
  · intro b hb
    set x : Int := (d.counts.ptr (b : Int)).addr with hx
    have hxcell : (DType.i32, x) ∈ cells d := ⟨b, hb, by rw [hx]⟩
    have hspan : Mem.span DType.i32 x ⊆ cellBytes (cells d) := span_sub_cellBytes hxcell
    have hfin : fin.read DType.i32 x = mA.read DType.i32 x :=
      Mem.read_congr (fun y hy => hAgree y (hspan hy))
    have h0 : init.read DType.i32 x = some (d.cntVal b) := by
      rw [hx]
      exact hinit b hb
    have hl : ∀ s ∈ G.map Prod.snd, MapsCell DType.i32 x (· + stepInc x s) s := by
      intro s hs
      obtain ⟨i, hi⟩ := mem_proj_of_mem_map G hs
      obtain ⟨m, hQset, -⟩ := hQ i
      exact mapsCell_stepInc hxcell (hQset.1 s hi)
    have hmA : mA.read DType.i32 x
        = some (d.cntVal b + ((G.map Prod.snd).map (stepInc x)).sum) :=
      replay_add (d := DType.i32) (a := x) (inc := stepInc x) (x := d.cntVal b) hrep hl h0
    have hterm : ∀ i : Fin (launch d).size,
        ((proj G i).map (stepInc x)).sum = IntV.ofNat (progCount d (i : Nat) b) := by
      intro i
      obtain ⟨m, hQset, -⟩ := hQ i
      exact hQset.2 b hb
    have hsum : ((G.map Prod.snd).map (stepInc x)).sum = IntV.ofNat (pairsIn d b) := by
      calc ((G.map Prod.snd).map (stepInc x)).sum
          = ∑ i, ((proj G i).map (stepInc x)).sum := sum_proj G (stepInc x)
        _ = ∑ i : Fin (launch d).size, IntV.ofNat (progCount d (i : Nat) b) :=
              Finset.sum_congr rfl (fun i _ => hterm i)
        _ = IntV.ofNat (∑ i : Fin (launch d).size, progCount d (i : Nat) b) :=
              ofNat_sum Finset.univ (fun i : Fin (launch d).size => progCount d (i : Nat) b)
        _ = IntV.ofNat (pairsIn d b) := by rw [pairsIn_eq_sum_progCount hd b]
    calc fin.read DType.i32 x
        = mA.read DType.i32 x := hfin
      _ = some (d.cntVal b + ((G.map Prod.snd).map (stepInc x)).sum) := hmA
      _ = some (d.cntVal b + IntV.ofNat (pairsIn d b)) := by rw [hsum]
  · intro z hz
    exact hframe z (by rw [cellBytes_cells]; exact hz)

end Claims.RouteHistogram
