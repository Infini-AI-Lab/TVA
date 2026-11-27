import Proof.Draft

/-!
# `combine_draft_tokens_i32`: after the logits indices (source lines 94-118)

The program loads its `seq_lens` and `prefill_len` entries; a request still in its prefill ends
there, the others write the last sampled token (`lastPart`, lines 101-105) and the drafts
(`draftPart`, `Proof.Draft`). `safe_tail` proves this part safe from any memory that keeps the
inputs and holds the request's logits indices.
-/

namespace Claims.CombineDraftTokens

open TVA

/-- Lines 102-105 of the kernel: the last sampled token, for the row `ri`, the prefill length
`pl`, the logits start `ls` and the first logit's position `fpos`. -/
def lastPart (d : Inst) (ri : I64) (pl ls fpos : I32) : Prog Unit :=
  if 0 < d.N then
    if pl.le fpos then do
      let last_token_id ← tl.load1 (d.last_sampled.ptr 0 + ri)
      let last32 ← tl.convert .i64 .i32 (Tensor.scalar last_token_id)
      tl.store1 (d.input_ids.ptr 0 + ls) last32.item
    else pure ()
  else pure ()

/-- Lines 94-118 of the kernel, for program `b`, the row `ri`, `cu_num_logits[b]` (`cs`),
`cu_num_logits[b + 1]` (`ce`) and the query end `qe`. -/
def tailProg (d : Inst) (b : Nat) (ri : I64) (cs ce qe : I32) : Prog Unit := do
  let seq_len ← tl.load1 (d.seq_lens.ptr 0 + (IntV.ofNat b : I32))
  let prefill_len ← tl.load1 (d.prefill_len.ptr 0 + ri)
  if seq_len.le prefill_len then
    return ()
  else do
    lastPart d ri prefill_len (qe - (ce - cs)) (seq_len - (ce - cs))
    draftPart d ri qe (ce - cs - (IntV.ofNat d.N : I32))

/-- A store to elements of `input_ids` keeps the inputs. -/
theorem keep_R_of_ids {d : Inst} (hd : d.ok) {m m' init : Mem} {T : Set Int}
    (hT : T ⊆ d.input_ids.bytes) (hfr : ∀ x, x ∉ T → m' x = m x)
    (hR : ∀ x ∈ R d, m x = init x) : ∀ x ∈ R d, m' x = init x := by
  intro x hx
  rw [hfr x fun hxT => ?_, hR x hx]
  obtain ⟨-, h1, h2, h3, h4, h5, h6, h7⟩ := hd.hids
  have hxi := hT hxT
  simp only [R, Set.mem_union] at hx
  rcases hx with (((((hx | hx) | hx) | hx) | hx) | hx) | hx
  · exact Set.disjoint_left.mp h1 hxi hx
  · exact Set.disjoint_left.mp h2 hxi hx
  · exact Set.disjoint_left.mp h3 hxi hx
  · exact Set.disjoint_left.mp h4 hxi hx
  · exact Set.disjoint_left.mp h5 hxi hx
  · exact Set.disjoint_left.mp h6 hxi hx
  · exact Set.disjoint_left.mp h7 hxi hx

/-- **The part after the logits indices is safe.** -/
theorem safe_tail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {b : Nat} (hb : b < d.B) (ri : I64) (cs ce qe : I32)
    (hri : ri.toInt = ridx d init b) (hcs : cs.toInt = cuAt d init b)
    (hce : ce.toInt = cuAt d init (b + 1)) (hqe : qe.toInt = qend d init b)
    {m : Mem} (hR : ∀ x ∈ R d, m x = init x) (hV : d.input_ids.Valid m)
    (hQL : QL d init b m) :
    Safe E (R d) (d.input_ids.elems (lastE d init b) ∪ d.input_ids.elems (draftE d init b))
      (fun _ m' => Q d init b m') m (tailProg d b ri cs ce qe) := by
  obtain ⟨hr0, hrpl, hrls, hs0, hNl, hlB, hq, -⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
  obtain ⟨hq0, hq1⟩ := qsl_bounds hpre hb
  have hB20 := block_le hd
  have hN1 := hd.hN
  have hb31 : b < 2 ^ 31 := lt_trans hb hd.hgrid
  have hnl : (ce - cs).toInt = nlog d init b :=
    IntV.toInt_sub hce hcs (sf32 (by simp only [nlog] at hNl; omega) (by simp only [nlog] at hlB ⊢; omega))
  unfold tailProg
  -- line 94
  rw [ptr_add d.seq_lens _ (IntV.I32.toInt_ofNat hb31)]
  refine safe_load1_buf d.seq_lens (lt_of_lt_of_le hb hd.hseq)
    (valid_keep hpre.2.2.2.2.1 (sub_R_seq d) hR)
    ((d.seq_lens.span_sub (lt_of_lt_of_le hb hd.hseq)).trans (sub_R_seq d)) fun sv hsv => ?_
  have hsv' : sv.toInt = seqAt d init b := by
    rw [read_keep (fun x hx => hR x (sub_R_seq d hx)) (lt_of_lt_of_le hb hd.hseq)] at hsv
    exact (rd32_of hsv).symm
  -- line 95
  have hk : ((ridx d init b).toNat : Int) = ridx d init b := Int.toNat_of_nonneg hr0
  have hri' : ri.toInt = (((ridx d init b).toNat : Nat) : Int) := by rw [hk, hri]
  have hpl_lt : (ridx d init b).toNat < d.prefill_len.len := by omega
  rw [ptr_add d.prefill_len _ hri']
  refine safe_load1_buf d.prefill_len hpl_lt (valid_keep hpre.2.2.2.2.2.1 (sub_R_pl d) hR)
    ((d.prefill_len.span_sub hpl_lt).trans (sub_R_pl d)) fun pv hpv => ?_
  have hpv' : pv.toInt = plAt d init b := by
    rw [read_keep (fun x hx => hR x (sub_R_pl d hx)) hpl_lt, hk] at hpv
    exact (rd32_of hpv).symm
  -- line 96
  by_cases hle : seqAt d init b ≤ plAt d init b
  · have hc : sv.le pv = true := by rw [IntV.le_eq, hsv', hpv']; exact decide_eq_true hle
    rw [ite_eq_left hc]
    exact safe_ret ⟨hQL, fun h => absurd h.1 (not_lt.mpr hle), fun h => absurd h (not_lt.mpr hle)⟩
  have hc : ¬ (sv.le pv = true) := by rw [IntV.le_eq, hsv', hpv']; simpa using hle
  rw [ite_eq_right hc]
  have hdec : Decode d init b := lt_of_not_ge hle
  have hnd : (ce - cs - (IntV.ofNat d.N : I32)).toInt = ndraft d init b :=
    IntV.toInt_sub hnl (IntV.I32.toInt_ofNat (by omega)) (sf32 (by omega) (by omega))
  -- the two parts write disjoint slots
  have hdisj : Disjoint (d.input_ids.elems (lastE d init b)) (d.input_ids.elems (draftE d init b)) := by
    refine d.input_ids.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro t ⟨hlast, ht⟩ ⟨-, j, -, ht'⟩
    have hN : d.N = 1 := by have := hlast.2.1; omega
    simp only [ndraft, hN] at ht'
    push_cast at ht'
    omega
  refine Safe.seq (W₁ := d.input_ids.elems (lastE d init b)) ?_ hdisj
  unfold lastPart
  -- the draft part, from the memory after the last sampled token
  have hdraft : ∀ m', (∀ x ∈ R d, m' x = init x) → d.input_ids.Valid m' → QL d init b m' →
      QA d init b m' →
      Safe E (R d) (d.input_ids.elems (draftE d init b)) (fun _ m'' => Q d init b m'') m'
        (draftPart d ri qe (ce - cs - (IntV.ofNat d.N : I32))) :=
    fun m' h1 h2 h3 h4 => safe_draft hd hpre E hb ri qe _ hri hqe hnd hdec h1 h2 h3 h4
  by_cases hN : 0 < d.N
  · rw [ite_eq_left hN]
    have hsb := (rd32_bounds d.seq_lens init b).2
    have hsb' : seqAt d init b < 2 ^ 31 := hsb
    have hfpos : (sv - (ce - cs)).toInt = seqAt d init b - nlog d init b :=
      IntV.toInt_sub hsv' hnl (sf32 (by omega) (by omega))
    by_cases hf : plAt d init b ≤ seqAt d init b - nlog d init b
    · have hc' : pv.le (sv - (ce - cs)) = true := by
        rw [IntV.le_eq, hpv', hfpos]; exact decide_eq_true hf
      rw [ite_eq_left hc']
      have hlast : LastOn d init b := ⟨hdec, hN, hf⟩
      -- line 104
      have hls_lt : (ridx d init b).toNat < d.last_sampled.len := by omega
      rw [ptr_add d.last_sampled _ hri']
      refine safe_load1_buf d.last_sampled hls_lt (valid_keep hpre.2.2.1 (sub_R_last d) hR)
        ((d.last_sampled.span_sub hls_lt).trans (sub_R_last d)) fun lv hlv => ?_
      refine safe_convert (fun _ => ConvRel.exists_i64_i32 _) fun l32 hl32 => ?_
      -- line 105
      have hk0 : 0 ≤ qend d init b - nlog d init b := by simp only [qend] at hq ⊢; omega
      have hkk : ((qend d init b - nlog d init b).toNat : Int) = qend d init b - nlog d init b :=
        Int.toNat_of_nonneg hk0
      have hqb : qend d init b < 2 ^ 31 := (rd32_bounds d.query_start_loc init ((b + 1 : Nat) : Int)).2
      have hlsv : (qe - (ce - cs)).toInt = (((qend d init b - nlog d init b).toNat : Nat) : Int) := by
        rw [hkk]; exact IntV.toInt_sub hqe hnl (sf32 (by omega) (by omega))
      have hlt : (qend d init b - nlog d init b).toNat < d.input_ids.len := by omega
      have hmem : (qend d init b - nlog d init b).toNat ∈ lastE d init b := ⟨hlast, hkk⟩
      refine safe_last ?_
      rw [ptr_add d.input_ids _ hlsv]
      refine safe_store1 (d.input_ids.writable hV hlt) (d.input_ids.span_sub_elems hmem)
        fun m2 hm2 hfr2 => safe_ret ?_
      have hT := d.input_ids.span_sub hlt
      refine hdraft m2 (keep_R_of_ids hd hT hfr2 hR) ?_ (QL_keep hpre hb (hd.hids.1.mono_left hT) hfr2 hQL) ?_
      · refine Buf.valid_of_agree hV fun x hx => ?_
        by_cases hxs : x ∈ Mem.span .i32 (d.input_ids.ptr (((qend d init b - nlog d init b).toNat : Nat) : Int)).addr
        · obtain ⟨-, hw⟩ := Mem.writable_of_read hm2
          set a := (d.input_ids.ptr (((qend d init b - nlog d init b).toNat : Nat) : Int)).addr
          simp only [Mem.span, Set.mem_ofPred_eq, DType.bytes] at hxs
          have := hw (x - a).toNat (by simp only [DType.bytes]; omega)
          rwa [Int.toNat_of_nonneg (by omega), add_sub_cancel] at this
        · rw [hfr2 x hxs]; exact hV.2.2.2 x hx
      · intro _
        refine ⟨l32.item, ?_, ?_⟩
        · rw [← hkk]; exact hm2
        · have h1 : l32.item.toInt = Int.bmod lv.toInt (2 ^ 32) := convRel_i64_i32_bmod (hl32 ())
          rw [read_keep (fun x hx => hR x (sub_R_last d hx)) hls_lt, hk] at hlv
          rw [h1, rdt64, rd64_of hlv]
    · have hc' : ¬ (pv.le (sv - (ce - cs)) = true) := by
        rw [IntV.le_eq, hpv', hfpos]; simpa using hf
      rw [ite_eq_right hc']
      exact safe_ret (hdraft m hR hV hQL fun h => absurd h.2.2 hf)
  · rw [ite_eq_right hN]
    exact safe_ret (hdraft m hR hV hQL fun h => absurd h.2.1 hN)

end Claims.CombineDraftTokens
