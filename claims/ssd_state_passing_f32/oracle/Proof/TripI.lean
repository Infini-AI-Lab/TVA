import Claim
import TVA.Proof
import Proof.Tail
import Proof.IntFacts
import Proof.Reads

/-!
# One trip of the loop with initial states

`safe_trip_I`: from the invariant before trip `t` (the state, the pointers, the counters, the
slots done), one trip is safe, reads only inputs, writes only slot `t + 1`, and restores the
invariant for `t + 1`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- **The end of a trip with initial states**, from the state and the decay it settled on. -/
theorem safe_restI {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hinit : d.hasInit = true) {pm h t : Nat} (hh : h < d.nheads)
    (ht : t < d.nchunks)
    {offs_m : Tensor [BL - 0] I32} {final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {c : I32} (hc : c.toInt = t) {m₁ : Mem}
    (hagree : ∀ x ∈ Rin d, m₁ x = init x) (hpres : Pres init m₁)
    {sptrs outp initp' : Tensor [BL - 0] (Ptr .f32)} {dAp : Ptr .f32}
    (hsp : ∀ k : Fin BL, sptrs.get1 k = d.states.ptr (stIdx d t h (pm * BL + k) : Int))
    (hdA : dAp = d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int))
    (hop : ∀ k : Fin BL, outp.get1 k = d.out.ptr (stIdx d (t + 1) h (pm * BL + k) : Int))
    (hdone : Done d init pm h t m₁)
    {new_states states' : Tensor [BL] F32} {e lg' : I32} {dA_cs' : F32}
    (he : e.toInt = (eReq d t : Int)) (hlg : lg'.toInt = ((t + pc d (eReq d t) : Nat) : Int))
    (hN : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d →
      (new_states.get1 k).isFinite = true ∧
        (new_states.get1 k).toReal = newV init d t h (pm * BL + k))
    (hS : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d → (states'.get1 k).isFinite = true ∧
      |(states'.get1 k).toReal - xRef init d h (pm * BL + k) t| ≤
        tolIn init d h (pm * BL + k) t)
    (hA : Vals init d → dA_cs'.isFinite = true ∧ |dA_cs'.toReal - dRef init d h t| ≤ dErr init d h t) :
    Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
      (fun st' m' => InvI d init pm h (t + 1) st' m') m₁
      (restI (IntV.ofNat d.nchunks) (IntV.ofNat d.dim) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d))
        (IntV.ofNat (sDc d)) BL offs_m final_ptrs c (Tensor.scalar true) new_states e states' initp'
        sptrs dAp outp lg' dA_cs') := by
  have hpresv : ∀ {dt : DType} (b : Buf dt), b.Valid init → b.Valid m₁ := fun b hb =>
    Buf.valid_of_agree hb fun x hx => hpres x (hb.2.2.2 x hx)
  have hvo : d.out.Valid m₁ := hpresv _ hpre.2.2.2.2.2.1
  have hvf : d.finalst.Valid m₁ := hpresv _ hpre.2.2.2.2.2.2.1
  unfold restI
  refine (safe_tail hd hP hh ht hvo hvf hc hoff hop hfp true dA_cs' states' new_states
    (fun k hk hV => ⟨(hS k hk hV).1, fun _ => (hS k hk hV).2⟩) (fun h' => absurd h' (by simp)) hN
    (fun _ hV => hA hV)
    (fun r => (r, initp', sptrs.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)),
      dAp + (IntV.ofNat (sDc d) : I32), outp.map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)),
      e, lg' + IntV.ofInt 1))).mono ?_ le_rfl
  rintro x m' ⟨r, rfl, hpost⟩
  obtain ⟨hlane, hread, hframe, hpr⟩ := hpost
  have hRd : ∀ y ∈ Rin d, m' y = init y := fun y hy => by
    rw [hframe y (fun hs => Set.disjoint_left.mp (slotW_disjoint_Rin hd hh (by omega)) hs hy)]
    exact hagree y hy
  have hbound : t + 1 + pc d (eReq d t) < 2 ^ 31 := by
    have h1 := hostN_lt hd hinit
    rw [hostN_eq hd] at h1
    have h2 := pc_mono (d := d) (show eReq d t ≤ d.nreq - 1 by
      have := (eReq_spec hd ht).1; omega)
    omega
  have hprev : prevReq d (t + 1) = eReq d t := by simp [prevReq]
  refine ⟨hRd, fun y hy => hpr y (hpres y hy), hlane, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  · show (lg' + IntV.ofInt 1 : I32).toInt = _
    rw [IntV.I32.add_nat hlg toInt_one (by omega), hprev]
    push_cast
    ring
  · refine Done.succ hd hh ht hdone hframe fun m hm => ?_
    obtain ⟨hm1, hm2, hm3⟩ := hm
    have hk : m - pm * BL < BL := by omega
    refine ⟨(r.get1 ⟨m - pm * BL, hk⟩), ?_, ?_⟩
    · have := hread ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [show pm * BL + (m - pm * BL) = m by omega] using this
    · have := hlane ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [show pm * BL + (m - pm * BL) = m by omega] using this

/-- **One trip with initial states** is safe: from the invariant before trip `t` it reads only
inputs, writes only slot `t + 1`, and restores the invariant for `t + 1`. -/
theorem safe_trip_I {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hinit : d.hasInit = true) {pm h t : Nat} (hh : h < d.nheads)
    (ht : t < d.nchunks)
    {offs_m : Tensor [BL - 0] I32} {seq_idx_ptr : Ptr .i32} {chunk_offsets_ptr : Ptr .i32}
    {initstates_ptr final_ptrs : Tensor [BL - 0] (Ptr .f32)}
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hseq : seq_idx_ptr = d.seqIdx.ptr 0) (hco : chunk_offsets_ptr = d.chunkOffs.ptr 0)
    (hip : ∀ k : Fin BL, initstates_ptr.get1 k = d.initst.ptr (inIdx d 0 h (pm * BL + k) : Int))
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    {c : I32} (hc : c.toInt = t) {st : StI BL} {m₁ : Mem} (hI : InvI d init pm h t st m₁) :
    Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
      (fun st' m' => InvI d init pm h (t + 1) st' m') m₁
      (tripI (IntV.ofNat d.nmeta) (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
        (IntV.ofNat d.chunk) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d))
        (IntV.ofNat 1) (IntV.ofNat (sSc d)) (IntV.ofNat 1) BL offs_m seq_idx_ptr chunk_offsets_ptr
        initstates_ptr final_ptrs c st) := by
  obtain ⟨states, initp, sptrs, dAp, outp, prev, lg⟩ := st
  obtain ⟨hagree, hpres, hstate, hsp, hdA, hop, hprev, hlg, hdone⟩ := hI
  simp only at hstate hsp hdA hop hprev hlg
  subst hdA
  have hch := chunk_pos hd
  have hch31 := chunk_lt hd
  have hdim31 : d.dim < 2 ^ 31 := by have := hd.dim_lt; omega
  have hpresv : ∀ {dt : DType} (b : Buf dt), b.Valid init → b.Valid m₁ := fun b hb =>
    Buf.valid_of_agree hb fun x hx => hpres x (hb.2.2.2 x hx)
  have hmask := mask_lane' (offs_m := offs_m) (pm := pm) hdim31 hoff
  unfold tripI
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
  have hkrc : ((!IntV.eq prev (IntV.ofNat (eReq d t))) = true) ↔ prevReq d t ≠ eReq d t := by
    rw [IntV.not_eq_eq, decide_eq_true_eq, hprev, he31]
    omega
  -- the end of a trip, whatever the state and decay it settled on
  have leaf : ∀ (initp' : Tensor [BL - 0] (Ptr .f32)) (st' : Tensor [BL] F32) (lg' : I32) (a : F32),
      lg'.toInt = ((t + pc d (eReq d t) : Nat) : Int) →
      (∀ k : Fin BL, pm * BL + k < d.dim → Vals init d → (st'.get1 k).isFinite = true ∧
        |(st'.get1 k).toReal - xRef init d h (pm * BL + k) t| ≤ tolIn init d h (pm * BL + k) t) →
      (Vals init d → a.isFinite = true ∧ |a.toReal - dRef init d h t| ≤ dErr init d h t) →
      Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
        (fun st' m' => InvI d init pm h (t + 1) st' m') m₁
        (restI (IntV.ofNat d.nchunks) (IntV.ofNat d.dim) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d))
          (IntV.ofNat (sDc d)) BL offs_m final_ptrs c (Tensor.scalar true) nw
          (IntV.ofNat (eReq d t)) st' initp' sptrs (d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int))
          outp lg' a) :=
    fun initp' st' lg' a h1 h2 h3 =>
      safe_restI hd hP hpre hinit hh ht hoff hfp hc hagree hpres hsp rfl hop hdone he31 h1 hN h2 h3
  by_cases hkr : prevReq d t ≠ eReq d t
  · -- a new request ends in this chunk: reload the state, and shorten the decay
    rw [ite_eq_left (hkrc.mpr hkr)]
    obtain ⟨hreset, he1, hdivq, hoffR, hge⟩ := reset_facts hd ht hkr
    have hoffR_lt := offR_lt hd ht hkr
    have hmulE : (IntV.ofNat (eReq d t) * IntV.ofNat (sSc d) : I32).toInt =
        ((eReq d t * sSc d : Nat) : Int) := by
      refine IntV.I32.mul_nat he31 (sSc_toInt hd) ?_
      have h1 := hd.init_lt hinit
      have h2 : eReq d t * sSc d ≤ (d.nreq - 1) * (d.nheads * d.dim) :=
        Nat.mul_le_mul_right _ (by omega)
      unfold sSc at *
      omega
    have hvi : d.initst.Valid m₁ := hpresv _ hpre.2.2.1
    have hip' : ∀ k : Fin BL,
        (Tensor.map (fun (q : Ptr .f32) => q + IntV.ofNat (eReq d t) * (IntV.ofNat (sSc d) : I32))
          initstates_ptr).get1 k = d.initst.ptr (inIdx d (eReq d t) h (pm * BL + k) : Int) := by
      intro k
      rw [get1_map_add, hip k, hmulE, Buf.ptr_offset]
      congr 1
      unfold inIdx sSc
      push_cast
      ring
    have hinlen : ∀ k : Fin BL, pm * BL + k < d.dim →
        inIdx d (eReq d t) h (pm * BL + k) < d.initst.len := fun k hk =>
      lt_of_lt_of_le (inIdx_lt hnreq hh hk) (hd.init_len hinit)
    refine safe_load_buf d.initst
      (g := fun i : Idx [BL - 0] => inIdx d (eReq d t) h (pm * BL + (i.1 : Nat)))
      (en := fun i : Idx [BL - 0] => decide (pm * BL + (i.1 : Nat) < d.dim))
      (addrs_lane d.initst hmask hip') ?_ hvi ?_ fun st' hst' hoth => ?_
    · intro i hi
      exact hinlen i.1 (of_decide_eq_true hi)
    · refine sub_Rin_initst ?_
      rintro j ⟨i, hi, rfl⟩
      exact hinlen i.1 (of_decide_eq_true hi)
    have hSst : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d →
        (st'.get1 k).isFinite = true ∧
          |(st'.get1 k).toReal - xRef init d h (pm * BL + k) t| ≤ tolIn init d h (pm * BL + k) t := by
      intro k hk hV
      have hr := hst' (k, ()) (decide_eq_true hk)
      rw [read_agree hagree (span_Rin_initst (hinlen k hk))] at hr
      obtain ⟨hf, hv⟩ := init_read hV hinit hnreq hh hk hr
      refine ⟨hf, ?_⟩
      have hx : xRef init d h (pm * BL + k) t = startV init d (eReq d t) h (pm * BL + k) := by
        simp [xRef, hreset]
      have ht0 : tolIn init d h (pm * BL + k) t = 0 := by simp [tolIn, hreset]
      rw [hx, ht0]
      have hv' : (st'.get1 k).toReal = startV init d (eReq d t) h (pm * BL + k) := hv
      rw [hv', sub_self, abs_zero]
    -- `seq_idx_chunk_start`
    have hstart := start_off_toInt hd hc ht
    have hstl := chunk_start_lt hd ht
    have hstartp : seq_idx_ptr + ((tl.iminimum (Tensor.scalar (c * IntV.ofNat d.chunk))
        (Tensor.scalar (IntV.ofNat d.seqlen))).item * IntV.ofNat 1 : I32) =
        d.seqIdx.ptr (t * d.chunk : Nat) := by
      rw [Ptr.add_int, hseq, IntV.I32.mul_nat hstart toInt_ofNat1 (by omega), Buf.ptr_offset]
      simp
    have hlenS' : t * d.chunk < d.seqIdx.len := lt_of_lt_of_le hstl hd.seq_len
    refine safe_load1 (v0 := IntV.ofNat (sReq d t)) ?_ ?_ ?_
    · rw [hstartp, read_agree hagree (span_Rin_seqIdx hlenS')]
      exact hMeta.1 _ hstl
    · rw [hstartp]; exact span_Rin_seqIdx hlenS'
    -- the logical chunk
    have hsreq : sReq d t ≤ eReq d t := sReq_le_eReq hd ht
    have hsr_lt : sReq d t < 2 ^ 31 := by omega
    have hes : (IntV.ofNat (eReq d t) - IntV.ofNat (sReq d t) : I32).toInt =
        ((eReq d t - sReq d t : Nat) : Int) :=
      IntV.I32.sub_nat he31 (IntV.I32.toInt_ofNat hsr_lt) hsreq (by omega)
    have hpcd := pc_chunk hd ht
    obtain ⟨hslot, hbound⟩ := logical_slot hd ht hkr
    have hN31 := hostN_lt hd hinit
    have hlg' : (lg + (IntV.ofNat (eReq d t) - IntV.ofNat (sReq d t)) : I32).toInt =
        ((t + pc d (eReq d t) : Nat) : Int) := by
      rw [IntV.I32.add_nat hlg hes (by omega)]
      congr 1
      omega
    have hnmeta : d.nmeta = hostN d := by rw [hd.nmeta_eq, ite_eq_left hinit]
    have hnm31 : d.nmeta < 2 ^ 31 := hd.nmeta_lt
    have hlenC : t + pc d (eReq d t) < d.chunkOffs.len :=
      lt_of_lt_of_le (by omega : t + pc d (eReq d t) < d.nmeta) hd.offs_len
    have hAco : d.chunkOffs.Addrs
        (tl.addrs (Tensor.scalar (chunk_offsets_ptr + (lg + (IntV.ofNat (eReq d t) - IntV.ofNat (sReq d t)))))
          (some ((Tensor.scalar (lg + (IntV.ofNat (eReq d t) - IntV.ofNat (sReq d t)))).ltS
            (IntV.ofNat d.nmeta))))
        (fun _ => t + pc d (eReq d t)) (fun _ => true) := by
      refine Buf.addrs_of_get d.chunkOffs (fun i => ?_) (fun i _ => ?_)
      · simp only [Tensor.get_ltS]
        show IntV.lt (lg + (IntV.ofNat (eReq d t) - IntV.ofNat (sReq d t))) (IntV.ofNat d.nmeta) = true
        rw [IntV.lt_eq, hlg', IntV.I32.toInt_ofNat hnm31]
        simp only [decide_eq_true_eq]
        omega
      · show chunk_offsets_ptr + _ = _
        rw [Ptr.add_int, hco, hlg', Buf.ptr_offset]
        simp
    refine safe_load_buf d.chunkOffs (g := fun _ => t + pc d (eReq d t)) (en := fun _ => true) hAco
      (fun _ _ => hlenC) (hpresv _ hpre.2.2.2.2.1) ?_ fun cl hcl hcoth => ?_
    · refine sub_Rin_chunkOffs ?_
      rintro j ⟨i, hi, rfl⟩
      exact hlenC
    have hcoff : cl.item = IntV.ofNat (offR d t) := by
      have hr := hcl () rfl
      rw [read_agree hagree (span_Rin_chunkOffs hlenC), hMeta.2 (t + pc d (eReq d t)) (by omega), hslot] at hr
      exact (Option.some.inj hr).symm
    rw [hcoff]
    have hlt0 : ((IntV.ofInt 0 : I32).lt (IntV.ofNat (offR d t)) = true) ↔ 0 < offR d t := by
      rw [IntV.lt_eq, decide_eq_true_eq, toInt_zero, IntV.I32.toInt_ofNat (by omega)]
      omega
    by_cases hsub : 0 < offR d t
    · rw [ite_eq_left (hlt0.mpr hsub)]
      have hA1 : (IntV.ofNat (offR d t) - IntV.ofInt 1 : I32).toInt = ((offR d t - 1 : Nat) : Int) :=
        IntV.I32.sub_nat (IntV.I32.toInt_ofNat (by omega)) toInt_one (by omega) (by omega)
      have hA2 : ((IntV.ofNat (offR d t) - IntV.ofInt 1) * IntV.ofNat 1 : I32).toInt =
          (((offR d t - 1) * 1 : Nat) : Int) := IntV.I32.mul_nat hA1 toInt_ofNat1 (by omega)
      have hB1 : (IntV.ofNat d.chunk - IntV.ofInt 1 : I32).toInt = ((d.chunk - 1 : Nat) : Int) :=
        IntV.I32.sub_nat (IntV.I32.toInt_ofNat hch31) toInt_one (by omega) hch31
      have hB2 : ((IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32).toInt =
          (((d.chunk - 1) * 1 : Nat) : Int) := IntV.I32.mul_nat hB1 toInt_ofNat1 (by omega)
      have hdiff : ((IntV.ofNat (offR d t) - IntV.ofInt 1) * IntV.ofNat 1 -
          (IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32).toInt =
          (((offR d t - 1) * 1 : Nat) : Int) - (((d.chunk - 1) * 1 : Nat) : Int) :=
        IntV.toInt_sub hA2 hB2 (IntV.I32.sfits (by omega) (by omega))
      have hptr : d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int) +
          ((IntV.ofNat (offR d t) - IntV.ofInt 1) * IntV.ofNat 1 -
            (IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32) =
          d.dA.ptr (dAIdx d h t (offR d t - 1) : Int) := by
        rw [Ptr.add_int, hdiff, Buf.ptr_offset]
        congr 1
        unfold dAIdx
        omega
      have hlenB : dAIdx d h t (offR d t - 1) < d.dA.len :=
        lt_of_lt_of_le (dAIdx_lt hh ht (by omega)) hd.dA_len
      have hAbd : d.dA.Addrs
          (tl.addrs (Tensor.scalar (d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int) +
            ((IntV.ofNat (offR d t) - IntV.ofInt 1) * IntV.ofNat 1 -
              (IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32)))
            (some ((Tensor.scalar ((IntV.ofInt (-1) : I32).lt (IntV.ofNat (offR d t) - IntV.ofInt 1))).and
              ((Tensor.scalar (IntV.ofNat (offR d t) : I32)).ltS (IntV.ofNat d.chunk)))))
          (fun _ => dAIdx d h t (offR d t - 1)) (fun _ => true) := by
        refine Buf.addrs_of_get d.dA (fun i => ?_) (fun i _ => ?_)
        · simp only [Tensor.get_and, Tensor.get_ltS]
          show (IntV.lt (IntV.ofInt (-1) : I32) (IntV.ofNat (offR d t) - IntV.ofInt 1) &&
            IntV.lt (IntV.ofNat (offR d t) : I32) (IntV.ofNat d.chunk)) = true
          rw [Bool.and_eq_true, IntV.lt_eq, IntV.lt_eq, hA1, IntV.I32.toInt_ofNat (by omega),
            IntV.I32.toInt_ofNat hch31, IntV.toInt_ofInt_neg_one (w := 32) (by norm_num)]
          simp only [decide_eq_true_eq]
          omega
        · show _ + _ = _
          exact hptr
      refine safe_load_buf d.dA (g := fun _ => dAIdx d h t (offR d t - 1)) (en := fun _ => true) hAbd
        (fun _ _ => hlenB) (hpresv _ hpre.2.1) ?_ fun bl hbl hbo => ?_
      · refine sub_Rin_dA ?_
        rintro j ⟨i, hi, rfl⟩
        exact hlenB
      have hb := hbl () rfl
      rw [read_agree hagree (span_Rin_dA hlenB)] at hb
      have hsubT : hasSub d t = true := by simp [hasSub, hreset, hsub]
      refine safe_sub fun a ha => ?_
      exact leaf _ st' _ (a.get ()) hlg' hSst (fun hV =>
        dA_ok hV hh ht hq hch (Or.inr ⟨hsubT, hb, ha ()⟩) (fun _ => by omega))
    · rw [ite_eq_right (fun h' => hsub (hlt0.mp h'))]
      have hns : hasSub d t = false := by simp [hasSub, hsub]
      exact leaf _ st' _ q hlg' hSst (fun hV =>
        dA_ok (b := q) hV hh ht hq hch (Or.inl ⟨hns, rfl⟩) (fun h' => by rw [hns] at h'; cases h'))
  · -- no reset: the state and the decay are the previous ones
    have heq : prevReq d t = eReq d t := not_not.mp hkr
    rw [ite_eq_right (fun h' => hkr (hkrc.mp h'))]
    refine leaf initp states lg q ?_ ?_ ?_
    · rw [hlg, heq]
    · intro k hk hV
      obtain ⟨hf, he⟩ := hstate k hk hV
      refine ⟨hf, ?_⟩
      obtain ⟨hx, ht'⟩ := xRef_of_eq (m₀ := init) (h := h) (m := pm * BL + k) hd ht heq
      rw [hx, ht']
      exact he
    · intro hV
      have hns := no_sub_of_eq hd ht heq
      exact dA_ok (b := q) hV hh ht hq hch (Or.inl ⟨hns, rfl⟩) (fun h' => by rw [hns] at h'; cases h')

end Claims.SsdStatePassing
