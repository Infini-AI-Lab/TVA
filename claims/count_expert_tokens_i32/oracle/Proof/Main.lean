import Proof.Loop

/-!
# The claim

Each program of the launch writes element `program_id` of `expert_num_tokens` and reads only the
ids (and the map). This file proves the body's safety with those footprints and puts the programs
together with the library's rule for independent programs.
-/

namespace Claims.CountExpertTokens

open TVA

/-! ## Unfolding the body -/

/-- The body of program `curr`, unfolded to its binds. -/
theorem bodyE_unfold (d : Inst) (curr : I32) :
    bodyE d curr =
      (tl.idiv (Tensor.scalar ((IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1)))
          (Tensor.scalar (IntV.ofNat d.B : I32)) >>= fun _ =>
        tl.idiv (Tensor.scalar ((IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1)))
          (Tensor.scalar (IntV.ofNat d.B : I32)) >>= fun trips =>
        tl.forRange (0 : I32) trips.item (1 : I32)
          (tl.zeros [d.B] .i32, d.ids.ptr 0 + tl.arange 0 d.B)
          (loopBody d.dtype d.hasMap d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr) >>= fun st =>
        if curr.lt (IntV.ofNat d.numLocal) then
          tl.store1 (d.out.ptr 0 + curr) (Tensor.foldAll (· + ·) (0 : I32) st.1)
        else pure ()) := rfl

/-! ## The scalar store's bytes -/

/-- A scalar access has the lane address of its pointer. -/
theorem addrs_scalar (dt : DType) (p : Ptr dt) :
    tl.addrs (Tensor.scalar p) none = (fun _ => some p.addr) := rfl

/-- `out + curr` points at element `e` of the output. -/
theorem out_ptr_eq {d : Inst} {e : Nat} (curr : I32) (hc : curr.toInt = (e : Int)) :
    (d.out.ptr 0 + curr) = d.out.ptr (e : Int) := by
  show (d.out.ptr 0).offset curr.toInt = d.out.ptr (e : Int)
  rw [Buf.ptr_offset, hc, Int.zero_add]

/-! ## The trip count, for a symbolic upper bound -/

/-- The loop bounds `0`, `hi`, `1` are covered by `tl.RangeOk`. -/
theorem rangeOk_hi {d : Inst} (hd : d.ok) {hi : I32} (hhi : hi.toInt = (numTrips d : Int)) :
    tl.RangeOk (0 : I32) hi (1 : I32) := by
  have h0 : (0 : I32).toInt = 0 := toInt_zero_i32
  have h1 : (1 : I32).toInt = 1 := toInt_one_i32
  have hir := intRange_32
  have hlt := numTrips_lt hd
  refine ⟨by rw [h1]; norm_num, ?_, ?_, ?_⟩
  · rw [hhi, h1, hir]
    omega
  · rw [h0, hir]
    omega
  · rw [h1, hir]
    norm_num

/-- The loop runs `numTrips d` times. -/
theorem tripCount_hi {d : Inst} {hi : I32}
    (hhi : hi.toInt = (numTrips d : Int)) :
    tl.tripCount ((0 : I32).toInt) hi.toInt ((1 : I32).toInt) = numTrips d := by
  have h0 : (0 : I32).toInt = 0 := toInt_zero_i32
  have h1 : (1 : I32).toInt = 1 := toInt_one_i32
  simp only [tl.tripCount, h0, h1, hhi, Int.ediv_one]
  rw [show (↑(numTrips d) - 0 + 1 - 1 : Int) = ↑(numTrips d) by ring, Int.toNat_natCast]

/-! ## The loop, with an arbitrary continuation -/

/-- The loop is safe and ends with every lane counting its positions, for any continuation that accepts that. -/
theorem safe_loop_k {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {e : Nat} {curr : I32} (hc : curr.toInt = (e : Int)) {hi : I32}
    (hhi : hi.toInt = (numTrips d : Int))
    (acc : Tensor [d.B] I32) (ptrs : Tensor [d.B] (Ptr d.dtype))
    (hacc0 : ∀ l : Fin d.B, (acc.get1 l).toInt = (cnt d e 0 l.val : Int))
    (hptr0 : ∀ l : Fin d.B, ptrs.get1 l = d.ids.ptr (tripG d 0 (l, ()) : Int))
    {Q : Unit → Mem → Prop}
    {k : Tensor [d.B] I32 × Tensor [d.B] (Ptr d.dtype) → Prog Unit}
    (hk : ∀ st m, m = init →
      (∀ l : Fin d.B, (st.1.get1 l).toInt = (cnt d e (numTrips d) l.val : Int)) →
      Safe E (Rin d) (Wout d e) Q m (k st)) :
    Safe E (Rin d) (Wout d e) Q init
      (tl.forRange (0 : I32) hi (1 : I32) (acc, ptrs)
        (loopBody d.dtype d.hasMap d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr) >>= k) := by
  have hmain : Safe E (Rin d)
      (iterW (fun _ => ∅) (tl.tripCount (0 : I32).toInt hi.toInt (1 : I32).toInt) ∪ Wout d e)
      Q init
      (tl.forRange (0 : I32) hi (1 : I32) (acc, ptrs)
        (loopBody d.dtype d.hasMap d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr) >>= k) := by
    refine safe_forRange (lo := (0 : I32)) (hi := hi) (step := (1 : I32)) (init := (acc, ptrs))
      (body := loopBody d.dtype d.hasMap d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr)
      (k := k) (hrange := rangeOk_hi hd hhi)
      (I := fun t st m => m = init ∧
        (∀ l : Fin d.B, (st.1.get1 l).toInt = (cnt d e t l.val : Int)) ∧
        (∀ l : Fin d.B, st.2.get1 l = d.ids.ptr (tripG d t (l, ()) : Int)))
      (Wt := fun _ => ∅) (Wk := Wout d e)
      (hWt := by simp) (hWk := by rw [iterW_empty]; exact Set.empty_disjoint (Wout d e))
      (h0 := ⟨rfl, hacc0, hptr0⟩) (hstep := ?_) (hk := ?_)
    · intro t ht st m₁ hI
      rw [tripCount_hi hhi] at ht
      obtain ⟨rfl, hacc, hptr⟩ := hI
      obtain ⟨a, b⟩ := st
      rw [tripArg, loopBody_eq]
      by_cases hm : d.hasMap = true
      · rw [hm]
        exact safe_trip_true hd hm hpre E ht curr hc a b hacc hptr
      · have hm' : d.hasMap = false := by cases h : d.hasMap <;> simp_all
        rw [hm']
        exact safe_trip_false hd hm' hpre E ht curr hc a b hacc hptr
    · intro st m₁ hI
      obtain ⟨hm, hacc, -⟩ := hI
      rw [tripCount_hi hhi] at hacc
      exact hk st m₁ hm hacc
  rw [iterW_empty, Set.empty_union] at hmain
  exact hmain

/-! ## The final store -/

/-- The guarded final store writes the sum of the lane counts at element `e`. -/
theorem safe_final {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {e : Nat} (he : e < d.numLocal) {curr : I32} (hc : curr.toInt = (e : Int))
    (acc : Tensor [d.B] I32)
    (hacc : ∀ l : Fin d.B, (acc.get1 l).toInt = (cnt d e (numTrips d) l.val : Int)) :
    Safe E (Rin d) (Wout d e) (fun _ m => Qout d init e m) init
      (if curr.lt (IntV.ofNat d.numLocal) then
        tl.store1 (d.out.ptr 0 + curr) (Tensor.foldAll (· + ·) (0 : I32) acc)
      else pure ()) := by
  have hguard : curr.lt (IntV.ofNat d.numLocal) = true := by
    rw [IntV.lt_eq, decide_eq_true_iff, hc]
    have h : (IntV.ofNat d.numLocal : I32).toInt = (d.numLocal : Int) :=
      IntV.I32.toInt_ofNat hd.numLocal_lt
    rw [h]
    exact_mod_cast he
  have hp : (d.out.ptr 0 + curr) = d.out.ptr (e : Int) := out_ptr_eq curr hc
  rw [ite_eq_left hguard, hp]
  refine safe_last ?_
  have hok : init.Writable .i32 (d.out.ptr (e : Int)).addr :=
    Buf.writable d.out hpre.2.1 (lt_of_lt_of_le he hd.out_len)
  have hW : Mem.span .i32 (d.out.ptr (e : Int)).addr ⊆ Wout d e :=
    Buf.span_sub_elems d.out (by simp)
  refine safe_store1 (W := Wout d e) hok hW ?_
  intro m' hread hframe
  exact safe_ret ⟨Tensor.foldAll (· + ·) (0 : I32) acc, foldAll_toInt hd hacc, hread, hframe⟩

/-! ## The whole body -/

/-- **One program** (expert `e`) is safe: it reads the ids and the map, writes element `e` of the output, and ends with its count there. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {e : Nat} (he : e < d.numLocal) {curr : I32} (hc : curr.toInt = (e : Int)) :
    Safe E (Rin d) (Wout d e) (fun _ m => Qout d init e m) init (bodyE d curr) := by
  rw [bodyE_unfold]
  have hX : 0 ≤ ((IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1)).toInt := by
    rw [sum_numel_B hd]
    exact Int.natCast_nonneg _
  have hY : 0 < (IntV.ofNat d.B : I32).toInt := by
    rw [B_val hd]
    exact_mod_cast B_pos hd
  refine safe_idiv1 (x := (IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1))
    (y := (IntV.ofNat d.B : I32)) hX hY ?_
  intro r hr
  refine safe_idiv1 (x := (IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1))
    (y := (IntV.ofNat d.B : I32)) hX hY ?_
  intro trips htrips
  have hhi : trips.item.toInt = (numTrips d : Int) := trips_val hd htrips
  refine safe_loop_k hd hpre E hc hhi (tl.zeros [d.B] .i32)
    (d.ids.ptr 0 + tl.arange 0 d.B) (acc_zero (d := d) e) (initPtrs hd) ?_
  intro st m hm hacc
  rw [hm]
  obtain ⟨a, b⟩ := st
  exact safe_final hd hpre E he hc a hacc

/-! ## Footprints

Element `e` of the output is one element of `d.out`, so the elements of distinct experts are
disjoint, and the output shares no byte with the ids or the map. -/

/-- A program's write footprint lies in the output's local elements. -/
theorem Wout_sub_elems {d : Inst} {i : Nat} (hi : i < d.numLocal) :
    Wout d i ⊆ d.out.elems {k | k < d.numLocal} := by
  rintro x ⟨k, hk, hx⟩
  rw [Set.mem_singleton_iff] at hk
  refine ⟨i, hi, ?_⟩
  rwa [hk] at hx

/-- Distinct programs write distinct elements. -/
theorem disjoint_Wout_Wout {d : Inst} {i j : Nat} (hij : i ≠ j) :
    Disjoint (Wout d i) (Wout d j) := by
  have hcast : (i : Int) ≠ (j : Int) := by exact_mod_cast hij
  rw [Set.disjoint_left]
  rintro x ⟨k, hk, hx⟩ ⟨k', hk', hx'⟩
  rw [Set.mem_singleton_iff] at hk hk'
  rw [hk] at hx
  rw [hk'] at hx'
  exact Set.disjoint_left.mp (Buf.span_disjoint d.out hcast) hx hx'

/-- No program writes what any program reads. -/
theorem disjoint_Wout_Rin {d : Inst} (hd : d.ok) {i : Nat} (hi : i < d.numLocal) :
    Disjoint (Wout d i) (Rin d) := by
  have hiB : i < d.out.len := lt_of_lt_of_le hi hd.out_len
  rw [Set.disjoint_left]
  intro x hxW hxR
  have hxB : x ∈ d.out.bytes :=
    Buf.elems_sub_bytes d.out
      (by
        intro k hk
        rw [Set.mem_singleton_iff] at hk
        subst hk
        exact hiB) hxW
  simp only [Rin] at hxR
  rcases hxR with hxids | hxmap
  · exact Set.disjoint_left.mp hd.out_ids hxB
      (Buf.elems_sub_bytes d.ids (by intro k hk; exact lt_of_lt_of_le hk hd.ids_len) hxids)
  · by_cases hm : d.hasMap = true
    · rw [ite_eq_left hm] at hxmap
      exact Set.disjoint_left.mp (hd.out_emap hm) hxB
        (Buf.elems_sub_bytes d.emap
          (by intro k hk; exact lt_of_lt_of_le hk (hd.emap_len hm)) hxmap)
    · rw [ite_eq_right hm] at hxmap
      exact hxmap

/-- A program writes nothing another reads or writes. -/
theorem disjoint_Wout_union {d : Inst} (hd : d.ok) {i j : Nat} (hij : i ≠ j)
    (hi : i < d.numLocal) :
    Disjoint (Wout d i) (Rin d ∪ Wout d j) := by
  rw [Set.disjoint_left]
  intro x hx hx'
  rcases hx' with hxR | hxW
  · exact Set.disjoint_left.mp (disjoint_Wout_Rin hd hi) hx hxR
  · exact Set.disjoint_left.mp (disjoint_Wout_Wout hij) hx hxW

/-! ## The claim -/

/-- A program index of the launch is a local expert. -/
theorem idx_lt_numLocal {d : Inst} (i : Fin (launch d).size) : (i : Nat) < d.numLocal :=
  lt_of_lt_of_eq i.isLt (launch_size d)

/-- **The claim.** Program `e` is safe on its own (`safe_program`) with the footprints `Rin` and
`Wout d e`; distinct programs write distinct elements of `out`, which no program reads. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => Rin d) (fun i => Wout d i.val)
    (fun i m => Qout d init i.val m) (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · have hi : (i : Nat) < d.numLocal := idx_lt_numLocal i
    have hc : (IntV.ofNat (i : Nat) : I32).toInt = ((i : Nat) : Int) :=
      IntV.I32.toInt_ofNat (lt_of_lt_of_le hi (le_of_lt hd.numLocal_lt))
    show Safe _ _ _ _ _ (bodyE d (tl.program_id ((launch d).ctx i) 0))
    rw [pid_eq d i hi]
    exact safe_program hd hpre ((launch d).env P) hi hc
  · intro i j hij
    have hne : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact disjoint_Wout_union hd hne (idx_lt_numLocal i)
  · intro i
    exact (disjoint_Wout_Rin hd (idx_lt_numLocal i)).symm
  · intro fin hloc hfr
    refine ⟨?_, ?_⟩
    · intro e he
      obtain ⟨m, ⟨c, hc, hread, -⟩, hframe⟩ := hloc ⟨e, (launch_size d).symm ▸ he⟩
      refine ⟨c, ?_, hc⟩
      rw [Mem.read_congr (m₂ := m)
        (fun x hx => hframe x (Or.inr (Buf.span_sub_elems d.out (S := {e}) (j := e) (by simp) hx)))]
      exact hread
    · intro z hz
      refine hfr z ?_
      intro i hzi
      exact hz (Wout_sub_elems (idx_lt_numLocal i) hzi)

end Claims.CountExpertTokens
