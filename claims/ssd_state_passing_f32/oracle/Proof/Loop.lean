import Claim
import TVA.Proof
import Proof.TripI
import Proof.TripN

/-!
# The chunk loop

The loop `tl.forRange(0, nchunks)` over the trips: by `safe_forRange` with the invariants `InvI` /
`InvN`, trip `t` writing slot `t + 1`. After the loop all slots are done.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The chunk loop `range(0, nchunks)` is within `tl.RangeOk`. -/
theorem rangeOk {d : Inst} (hd : d.ok) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat d.nchunks : I32) (IntV.ofInt 1 : I32) := by
  have hn := nchunks_lt hd
  have hnn : (IntV.ofNat d.nchunks : I32).toInt = d.nchunks := IntV.I32.toInt_ofNat hn
  unfold tl.RangeOk
  rw [hnn, toInt_zero, toInt_one]
  simp only [IntV.maxInt]
  push_cast
  refine ⟨by norm_num, by omega, by norm_num, by norm_num⟩

/-- The chunk loop runs `nchunks` trips. -/
theorem tripCount_eq {d : Inst} (hd : d.ok) :
    tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat d.nchunks : I32).toInt
      (IntV.ofInt 1 : I32).toInt = d.nchunks := by
  have hnn : (IntV.ofNat d.nchunks : I32).toInt = d.nchunks := IntV.I32.toInt_ofNat (nchunks_lt hd)
  unfold tl.tripCount
  rw [hnn, toInt_zero, toInt_one]
  simp

/-- The loop's induction variable at trip `t` is `t`. -/
theorem loopVar_toInt (t : Nat) (ht : t < 2 ^ 31) :
    (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt =
      (t : Int) := by
  rw [toInt_zero, toInt_one]
  have : ((0 : Nat) : Int) + (t : Int) * ((1 : Nat) : Int) = (t : Int) := by simp
  rw [this]
  exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩

/-- **The loop with initial states**: from the invariant at trip `0`, it is safe, writes slots
`1..nchunks`, and ends with all slots done. -/
theorem safe_loop_I {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hinit : d.hasInit = true) {pm h : Nat} (hh : h < d.nheads)
    {offs_m : Tensor [BL - 0] I32} {seq_idx_ptr : Ptr .i32} {chunk_offsets_ptr : Ptr .i32}
    {initstates_ptr final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hseq : seq_idx_ptr = d.seqIdx.ptr 0) (hco : chunk_offsets_ptr = d.chunkOffs.ptr 0)
    (hip : ∀ k : Fin BL, initstates_ptr.get1 k = d.initst.ptr (inIdx d 0 h (pm * BL + k) : Int))
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {st0 : StI BL} {m : Mem} (h0 : InvI d init pm h 0 st0 m) :
    Safe ((launch d).env P) (Rin d) (iterW (fun t => slotW d h pm (t + 1)) d.nchunks)
      (fun _ m' => Done d init pm h d.nchunks m') m
      (tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.nchunks : I32) (IntV.ofInt 1 : I32) st0
        (tripI (IntV.ofNat d.nmeta) (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
          (IntV.ofNat d.chunk) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d))
          (IntV.ofNat 1) (IntV.ofNat (sSc d)) (IntV.ofNat 1) BL offs_m seq_idx_ptr chunk_offsets_ptr
          initstates_ptr final_ptrs) >>= fun _ => pure ()) := by
  have hcount := tripCount_eq hd
  have hsafe := safe_forRange (E := (launch d).env P) (R := Rin d) (m := m)
    (Q := fun (_ : Unit) m' => Done d init pm h d.nchunks m')
    (body := tripI (IntV.ofNat d.nmeta) (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
          (IntV.ofNat d.chunk) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d))
          (IntV.ofNat 1) (IntV.ofNat (sSc d)) (IntV.ofNat 1) BL offs_m seq_idx_ptr chunk_offsets_ptr
          initstates_ptr final_ptrs)
    (k := fun _ => pure ()) (rangeOk hd) (fun t st m' => InvI d init pm h t st m')
    (fun t => slotW d h pm (t + 1)) (Wk := ∅) ?_ (Set.disjoint_empty _) h0 ?_ ?_
  · rw [hcount] at hsafe
    simpa using hsafe
  · intro t u htu hu
    rw [hcount] at hu
    exact slotW_disjoint hd hh (by omega) (by omega) (by omega)
  · intro t ht st m₁ hI
    rw [hcount] at ht
    exact safe_trip_I hd hP hpre hinit hh ht hoff hseq hco hip hfp
      (loopVar_toInt t (by have := nchunks_lt hd; omega)) hI
  · intro st m₁ hI
    rw [hcount] at hI
    exact safe_ret hI.2.2.2.2.2.2.2.2

/-- **The loop without initial states**. -/
theorem safe_loop_N {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hnoinit : d.hasInit = false) {pm h : Nat} (hh : h < d.nheads)
    {offs_m : Tensor [BL - 0] I32} {seq_idx_ptr : Ptr .i32}
    {final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hseq : seq_idx_ptr = d.seqIdx.ptr 0)
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {st0 : StN BL} {m : Mem} (h0 : InvN d init pm h 0 st0 m) :
    Safe ((launch d).env P) (Rin d) (iterW (fun t => slotW d h pm (t + 1)) d.nchunks)
      (fun _ m' => Done d init pm h d.nchunks m') m
      (tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.nchunks : I32) (IntV.ofInt 1 : I32) st0
        (tripN (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen) (IntV.ofNat d.chunk)
          (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d)) (IntV.ofNat 1) BL offs_m
          seq_idx_ptr final_ptrs) >>= fun _ => pure ()) := by
  have hcount := tripCount_eq hd
  have hsafe := safe_forRange (E := (launch d).env P) (R := Rin d) (m := m)
    (Q := fun (_ : Unit) m' => Done d init pm h d.nchunks m')
    (body := tripN (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen) (IntV.ofNat d.chunk)
          (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d)) (IntV.ofNat 1) BL offs_m
          seq_idx_ptr final_ptrs)
    (k := fun _ => pure ()) (rangeOk hd) (fun t st m' => InvN d init pm h t st m')
    (fun t => slotW d h pm (t + 1)) (Wk := ∅) ?_ (Set.disjoint_empty _) h0 ?_ ?_
  · rw [hcount] at hsafe
    simpa using hsafe
  · intro t u htu hu
    rw [hcount] at hu
    exact slotW_disjoint hd hh (by omega) (by omega) (by omega)
  · intro t ht st m₁ hI
    rw [hcount] at ht
    exact safe_trip_N hd hP hpre hnoinit hh ht hoff hseq hfp
      (loopVar_toInt t (by have := nchunks_lt hd; omega)) hI
  · intro st m₁ hI
    rw [hcount] at hI
    exact safe_ret hI.2.2.2.2.2.2.2

end Claims.SsdStatePassing
