import Claim
import TVA.Proof
import Proof.Tail
import Proof.IntFacts
import Proof.Reads

/-!
# One trip of the loop without initial states

`safe_trip_N`: from the invariant before trip `t`, one trip is safe, reads only inputs, writes only
slot `t + 1`, and restores the invariant for `t + 1`. The scale is `0` where the request at the
chunk's end changes.
-/
namespace Claims.SsdStatePassing

open TVA

/-- **The end of a trip without initial states**, from the state and the decay it settled on. -/
theorem safe_restN {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {pm h t : Nat} (hh : h < d.nheads) (ht : t < d.nchunks)
    {offs_m : Tensor [BL - 0] I32} {final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {c : I32} (hc : c.toInt = t) {m₁ : Mem}
    (hagree : ∀ x ∈ Rin d, m₁ x = init x) (hpres : Pres init m₁)
    {sptrs outp : Tensor [BL - 0] (Ptr .f32)} {dAp : Ptr .f32}
    (hsp : ∀ k : Fin BL, sptrs.get1 k = d.states.ptr (stIdx d t h (pm * BL + k) : Int))
    (hdA : dAp = d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int))
    (hop : ∀ k : Fin BL, outp.get1 k = d.out.ptr (stIdx d (t + 1) h (pm * BL + k) : Int))
    (hdone : Done d init pm h t m₁)
    {new_states states' : Tensor [BL] F32} {e : I32} {dA_cs' : F32} (sm : Bool)
    (he : e.toInt = (eReq d t : Int))
    (hN : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d →
      (new_states.get1 k).isFinite = true ∧
        (new_states.get1 k).toReal = newV init d t h (pm * BL + k))
    (hS : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d → (states'.get1 k).isFinite = true ∧
      (sm = true → |(states'.get1 k).toReal - xRef init d h (pm * BL + k) t| ≤
        tolIn init d h (pm * BL + k) t))
    (hZ : sm = false → ∀ k : Fin BL, pm * BL + k < d.dim →
      xRef init d h (pm * BL + k) t = 0 ∧ tolIn init d h (pm * BL + k) t = 0)
    (hA : sm = true → Vals init d → dA_cs'.isFinite = true ∧
      |dA_cs'.toReal - dRef init d h t| ≤ dErr init d h t) :
    Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
      (fun st' m' => InvN d init pm h (t + 1) st' m') m₁
      (restN (IntV.ofNat d.nchunks) (IntV.ofNat d.dim) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d))
        (IntV.ofNat (sDc d)) BL offs_m final_ptrs c (Tensor.scalar sm) new_states e states'
        sptrs dAp outp dA_cs') := by
  have hpresv : ∀ {dt : DType} (b : Buf dt), b.Valid init → b.Valid m₁ := fun b hb =>
    Buf.valid_of_agree hb fun x hx => hpres x (hb.2.2.2 x hx)
  have hvo : d.out.Valid m₁ := hpresv _ hpre.2.2.2.2.2.1
  have hvf : d.finalst.Valid m₁ := hpresv _ hpre.2.2.2.2.2.2.1
  unfold restN
  refine (safe_tail hd hP hh ht hvo hvf hc hoff hop hfp sm dA_cs' states' new_states hS hZ hN hA
    (fun r => (r, sptrs.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)),
      dAp + (IntV.ofNat (sDc d) : I32),
      outp.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)), e))).mono ?_ le_rfl
  rintro x m' ⟨r, rfl, hpost⟩
  obtain ⟨hlane, hread, hframe, hpr⟩ := hpost
  have hRd : ∀ y ∈ Rin d, m' y = init y := fun y hy => by
    rw [hframe y (fun hs => Set.disjoint_left.mp (slotW_disjoint_Rin hd hh (by omega)) hs hy)]
    exact hagree y hy
  have hprev : prevReq d (t + 1) = eReq d t := by simp [prevReq]
  refine ⟨hRd, fun y hy => hpr y (hpres y hy), hlane, ?_, ?_, ?_, ?_, ?_⟩
  · intro k
    show (sptrs.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32))).get1 k = _
    rw [get1_map_add, hsp k, sSc_toInt hd, Buf.ptr_offset]
    congr 1
    unfold stIdx sSc
    push_cast
    ring
  · show dAp + (IntV.ofNat (sDc d) : I32) = _
    rw [Ptr.add_int, hdA, sDc_toInt hd, Buf.ptr_offset]
    congr 1
    unfold dAIdx sDc
    push_cast
    ring
  · intro k
    show (outp.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32))).get1 k = _
    rw [get1_map_add, hop k, sSc_toInt hd, Buf.ptr_offset]
    congr 1
    unfold stIdx sSc
    push_cast
    ring
  · show e.toInt = _
    rw [he, hprev]
  · refine Done.succ hd hh ht hdone hframe fun m hm => ?_
    obtain ⟨hm1, hm2, hm3⟩ := hm
    have hk : m - pm * BL < BL := by omega
    refine ⟨(r.get1 ⟨m - pm * BL, hk⟩), ?_, ?_⟩
    · have := hread ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [show pm * BL + (m - pm * BL) = m by omega] using this
    · have := hlane ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [show pm * BL + (m - pm * BL) = m by omega] using this

/-- **One trip without initial states** is safe: from the invariant before trip `t` it reads only
inputs, writes only slot `t + 1`, and restores the invariant for `t + 1`. -/
theorem safe_trip_N {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hnoinit : d.hasInit = false) {pm h t : Nat} (hh : h < d.nheads)
    (ht : t < d.nchunks)
    {offs_m : Tensor [BL - 0] I32} {seq_idx_ptr : Ptr .i32}
    {final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hseq : seq_idx_ptr = d.seqIdx.ptr 0)
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {c : I32} (hc : c.toInt = t) {st : StN BL} {m₁ : Mem} (hI : InvN d init pm h t st m₁) :
    Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
      (fun st' m' => InvN d init pm h (t + 1) st' m') m₁
      (tripN (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen) (IntV.ofNat d.chunk)
        (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d)) (IntV.ofNat 1) BL offs_m
        seq_idx_ptr final_ptrs c st) := by
  obtain ⟨states, sptrs, dAp, outp, prev⟩ := st
  obtain ⟨hagree, hpres, hstate, hsp, hdA, hop, hprev, hdone⟩ := hI
  simp only at hstate hsp hdA hop hprev
  subst hdA
  have hch := chunk_pos hd
  have hdim31 : d.dim < 2 ^ 31 := by have := hd.dim_lt; omega
  have hpresv : ∀ {dt : DType} (b : Buf dt), b.Valid init → b.Valid m₁ := fun b hb =>
    Buf.valid_of_agree hb fun x hx => hpres x (hb.2.2.2 x hx)
  have hmask := mask_lane' (offs_m := offs_m) (pm := pm) hdim31 hoff
  unfold tripN
  dsimp only
  -- `new_states = tl.load(states_ptrs, ...)`
  refine safe_load_buf d.states (g := fun i : Idx [BL - 0] => stIdx d t h (pm * BL + (i.1 : Nat)))
    (en := fun i : Idx [BL - 0] => decide (pm * BL + (i.1 : Nat) < d.dim))
    (addrs_lane d.states hmask hsp) ?_ (hpresv _ hpre.1) ?_ fun nw hnw hother => ?_
  · intro i hi
    exact lt_of_lt_of_le (stIdx_lt ht hh (of_decide_eq_true hi)) hd.states_len
  · refine sub_Rin_states ?_
    rintro j ⟨i, hi, rfl⟩
    exact lt_of_lt_of_le (stIdx_lt ht hh (of_decide_eq_true hi)) hd.states_len
  have hN : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d →
      (nw.get1 k).isFinite = true ∧ (nw.get1 k).toReal = newV init d t h (pm * BL + k) := by
    intro k hk hV
    have hr := hnw (k, ()) (decide_eq_true hk)
    have hlt := lt_of_lt_of_le (stIdx_lt ht hh hk) hd.states_len
    rw [read_agree hagree (span_Rin_states hlt)] at hr
    exact states_read hV ht hh hk hr
  -- `dA_cs = tl.load(dA_cs_ptr)`
  have hlenA : dAIdx d h t (d.chunk - 1) < d.dA.len :=
    lt_of_lt_of_le (dAIdx_lt hh ht (by omega)) hd.dA_len
  refine safe_load1_buf d.dA hlenA (hpresv _ hpre.2.1) (span_Rin_dA hlenA) fun q hq => ?_
  rw [read_agree hagree (span_Rin_dA hlenA)] at hq
  -- `seq_idx_chunk_end = tl.load(seq_idx_ptr + seq_end_off)`
  have hend := end_off_toInt hd hc ht
  have hetok := (endTok_facts hd ht).2.2
  have hsl31 := seqlen_lt hd
  have hMeta := hpre.2.2.2.2.2.2.2
  have hendp : seq_idx_ptr + (((tl.iminimum (Tensor.scalar ((c + IntV.ofInt 1) * IntV.ofNat d.chunk))
      (Tensor.scalar (IntV.ofNat d.seqlen))).item - IntV.ofInt 1) * IntV.ofNat 1 : I32) =
      d.seqIdx.ptr (endTok d t : Nat) := by
    rw [Ptr.add_int, hseq, IntV.I32.mul_nat hend toInt_ofNat1 (by omega), Buf.ptr_offset]
    simp
  have hlenS : endTok d t < d.seqIdx.len := lt_of_lt_of_le hetok hd.seq_len
  refine safe_load1 (v0 := IntV.ofNat (eReq d t)) ?_ ?_ ?_
  · rw [hendp, read_agree hagree (span_Rin_seqIdx hlenS)]
    exact hMeta.1 _ hetok
  · rw [hendp]; exact span_Rin_seqIdx hlenS
  have hnreq : eReq d t < d.nreq := (eReq_spec hd ht).1
  have hnreq8 := hd.nreq_le
  have he31 : (IntV.ofNat (eReq d t) : I32).toInt = (eReq d t : Int) :=
    IntV.I32.toInt_ofNat (by omega)
  have hsm : IntV.eq (IntV.ofNat (eReq d t)) prev = true ↔ prevReq d t = eReq d t := by
    rw [IntV.eq_eq, decide_eq_true_eq, hprev, he31]
    omega
  refine safe_restN hd hP hpre hh ht hoff hfp hc hagree hpres hsp rfl hop hdone
    (IntV.eq (IntV.ofNat (eReq d t)) prev) he31 hN ?_ ?_ ?_
  · intro k hk hV
    obtain ⟨hf, he⟩ := hstate k hk hV
    refine ⟨hf, fun hs => ?_⟩
    have heq := hsm.mp hs
    obtain ⟨hx, ht'⟩ := xRef_of_eq (m₀ := init) (h := h) (m := pm * BL + k) hd ht heq
    rw [hx, ht']
    exact he
  · intro hs k hk
    have hne : prevReq d t ≠ eReq d t := fun h' => by
      have := hsm.mpr h'
      rw [this] at hs
      cases hs
    obtain ⟨hreset, -, -, -, -⟩ := reset_facts hd ht hne
    refine ⟨?_, ?_⟩
    · simp [xRef, hreset, startV, hnoinit]
    · simp [tolIn, hreset]
  · intro hs hV
    have heq := hsm.mp hs
    have hns := no_sub_of_eq hd ht heq
    exact dA_ok (b := q) hV hh ht hq hch (Or.inl ⟨hns, rfl⟩) (fun h' => by rw [hns] at h'; cases h')

end Claims.SsdStatePassing
