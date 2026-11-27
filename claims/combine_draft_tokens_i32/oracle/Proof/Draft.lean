import Proof.Basic

/-!
# `combine_draft_tokens_i32`: the draft tokens (source lines 108-118)

A program past its prefill copies its `ndraft` draft tokens, truncated to `int32`, into the
`input_ids` slots `qend - ndraft + j`. `safe_draft` proves this part safe from any memory that
keeps the inputs, given what the program has written before (`QL`, `QA`), which it keeps.
-/

namespace Claims.CombineDraftTokens

open TVA

/-- Lines 108-118 of the kernel, for the request-state row `ri`, the query end `qe` and the
number of drafts `nd`. -/
def draftPart (d : Inst) (ri : I64) (qe nd : I32) : Prog Unit :=
  if (0 : I32).lt nd then do
    let mask := (tl.arange 0 d.BLOCK).ltS nd
    let stride64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.dstride : I32))).item
    let draft_tokens ← tl.load (d.draft_tokens.ptr 0 + ri * stride64 + tl.arange 0 d.BLOCK)
      (some mask)
    let draft32 ← tl.convert .i64 .i32 draft_tokens
    tl.store (d.input_ids.ptr 0 + qe - nd + tl.arange 0 d.BLOCK) draft32 (some mask)
  else pure ()

/-- `QL` depends only on the bytes of `logits_indices`. -/
theorem QL_keep {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.B) {m m' : Mem}
    {T : Set Int} (hT : Disjoint T d.logits_indices.bytes) (hfr : ∀ x, x ∉ T → m' x = m x)
    (h : QL d init b m) : QL d init b m' := by
  intro j hj
  obtain ⟨w, hw, hv⟩ := h j hj
  refine ⟨w, ?_, hv⟩
  obtain ⟨hc0, hc1⟩ := cu_bounds hpre hb
  have hnl : nlog d init b = cuAt d init (b + 1) - cuAt d init b := rfl
  have hk : ((cuAt d init b + j).toNat : Int) = cuAt d init b + j := Int.toNat_of_nonneg (by omega)
  have hlt : (cuAt d init b + j).toNat < d.logits_indices.len := by omega
  rw [← hk] at hw ⊢
  rw [← hw]
  refine Mem.read_congr fun x hx => hfr x fun hxT => ?_
  exact Set.disjoint_left.mp hT hxT (d.logits_indices.span_sub hlt hx)

/-- **The draft part is safe.** From a memory that keeps the inputs and holds the request's
logits indices and last sampled token, it writes the request's drafts and keeps the rest. -/
theorem safe_draft {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {b : Nat} (hb : b < d.B) (ri : I64) (qe nd : I32)
    (hri : ri.toInt = ridx d init b) (hqe : qe.toInt = qend d init b)
    (hnd : nd.toInt = ndraft d init b) (hdec : Decode d init b)
    {m : Mem} (hR : ∀ x ∈ R d, m x = init x) (hV : d.input_ids.Valid m)
    (hQL : QL d init b m) (hQA : QA d init b m) :
    Safe E (R d) (d.input_ids.elems (draftE d init b)) (fun _ m' => Q d init b m') m
      (draftPart d ri qe nd) := by
  obtain ⟨hr0, -, -, -, hNl, hlB, hq, hdr⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
  obtain ⟨hq0, hq1⟩ := qsl_bounds hpre hb
  have hB20 := block_le hd
  have hN1 := hd.hN
  unfold draftPart
  by_cases hpos : 0 < ndraft d init b
  · have hc : (0 : I32).lt nd = true := by
      rw [IntV.lt_eq, toInt_zero32, hnd]; exact decide_eq_true hpos
    rw [ite_eq_left hc]
    have hnd_le : ndraft d init b ≤ d.BLOCK := by simp only [ndraft]; omega
    have hnd_def : ndraft d init b = nlog d init b - d.N := rfl
    -- the stride
    refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun st hst => ?_
    have hst' : st.item.toInt = (d.dstride : Int) := by
      have := ConvRel.toInt_i32_i64 (hst ())
      simp only [Tensor.scalar, Tensor.item] at this ⊢
      rw [this]; exact IntV.I32.toInt_ofNat hd.hstride
    -- the draft row
    have hrow := hdr hpos
    have hdv : d.draft_tokens.Valid m := valid_keep hpre.2.2.2.2.2.2.1 (sub_R_draft d) hR
    have hlen := len_lt_of_valid64 hpre.2.2.2.2.2.2.1
    have hprod0 : 0 ≤ ridx d init b * d.dstride := by positivity
    have hmul : (ri * st.item).toInt = ridx d init b * d.dstride :=
      IntV.toInt_mul hri hst' (sf64 (by omega) (by omega))
    let en : Idx [d.BLOCK - 0] → Bool := fun i => decide (((i.1 : Nat) : Int) < ndraft d init b)
    have hen : ∀ i, en i = true → ((i.1 : Nat) : Int) < ndraft d init b := fun i h =>
      of_decide_eq_true h
    have hmask : ∀ i, ((tl.arange 0 d.BLOCK).ltS nd).get i = en i := by
      intro i
      rw [Tensor.get_ltS, IntV.lt_eq, arange_toInt (by omega), hnd]
    let gR : Idx [d.BLOCK - 0] → Nat := fun i => (ridx d init b * d.dstride + (i.1 : Nat)).toNat
    have hgR : ∀ i, (gR i : Int) = ridx d init b * d.dstride + (i.1 : Nat) := fun i =>
      Int.toNat_of_nonneg (by positivity)
    have hinR : ∀ i, en i = true → gR i < d.draft_tokens.len := by
      intro i he; have := hen i he; have := hgR i; omega
    have hAR : d.draft_tokens.Addrs
        (tl.addrs (d.draft_tokens.ptr 0 + ri * st.item + tl.arange 0 d.BLOCK) (some
          ((tl.arange 0 d.BLOCK).ltS nd))) gR en := by
      refine Buf.addrs_of_get _ hmask fun i _ => ?_
      rw [Tensor.get_ptrAdd, ptr_add d.draft_tokens _ hmul, Buf.ptr_offset,
        arange_toInt (by omega), hgR i]
    refine safe_load_buf d.draft_tokens hAR hinR hdv
      ((d.draft_tokens.elems_sub_bytes (by rintro j ⟨i, he, rfl⟩; exact hinR i he)).trans
        (sub_R_draft d))
      fun dv hdv' _ => ?_
    refine safe_convert (fun _ => ConvRel.exists_i64_i32 _) fun d32 hd32 => ?_
    -- the store
    let gW : Idx [d.BLOCK - 0] → Nat := fun i =>
      (qend d init b - ndraft d init b + (i.1 : Nat)).toNat
    have hgW : ∀ i, (gW i : Int) = qend d init b - ndraft d init b + (i.1 : Nat) := fun i =>
      Int.toNat_of_nonneg (by have := (i.1 : Nat).cast_nonneg (α := Int); omega)
    have hinW : ∀ i, en i = true → gW i < d.input_ids.len := by
      intro i he; have := hen i he; have := hgW i; omega
    have hsub : d.input_ids.ptr 0 + qe - nd = d.input_ids.ptr (qend d init b - ndraft d init b) := by
      rw [Ptr.sub_int_of_signed _ _ (by rw [hnd]; omega), ptr_add d.input_ids _ hqe,
        Buf.ptr_offset, hnd]
      ring_nf
    have hAW : d.input_ids.Addrs
        (tl.addrs (d.input_ids.ptr 0 + qe - nd + tl.arange 0 d.BLOCK) (some
          ((tl.arange 0 d.BLOCK).ltS nd))) gW en := by
      refine Buf.addrs_of_get _ hmask fun i _ => ?_
      rw [Tensor.get_ptrAdd, hsub, Buf.ptr_offset, arange_toInt (by omega), hgW i]
    have hinj : ∀ i j, en i = true → en j = true → gW i = gW j → i = j := by
      intro i j _ _ h
      have h' := congrArg (fun n : Nat => (n : Int)) h
      simp only [hgW] at h'
      exact Prod.ext (Fin.ext (by exact_mod_cast (by linarith : ((i.1 : Nat) : Int) = (j.1 : Nat))))
        rfl
    have himg : Buf.img gW en ⊆ draftE d init b := by
      rintro t ⟨i, he, rfl⟩
      exact ⟨hdec, (i.1 : Nat), hen i he, hgW i⟩
    refine safe_last ?_
    refine safe_store_buf d.input_ids hAW hinW hinj hV (d.input_ids.elems_mono himg)
      fun m' hm' hfr _ => safe_ret ?_
    have hT : d.input_ids.elems (Buf.img gW en) ⊆ d.input_ids.bytes :=
      d.input_ids.elems_sub_bytes (by rintro j ⟨i, he, rfl⟩; exact hinW i he)
    refine ⟨QL_keep hpre hb (hd.hids.1.mono_left hT) hfr hQL, ?_, ?_⟩
    · -- the last sampled token's slot is not a draft slot
      intro hlast
      obtain ⟨w, hw, hwv⟩ := hQA hlast
      refine ⟨w, ?_, hwv⟩
      have hN : d.N = 1 := by have := hlast.2.1; omega
      have hk0 : 0 ≤ qend d init b - nlog d init b := by
        have := hpre.2.2.2.2.2.2.2.2.2.1 b hb; simp only [qend] at hq ⊢; omega
      have hk : ((qend d init b - nlog d init b).toNat : Int) = qend d init b - nlog d init b :=
        Int.toNat_of_nonneg hk0
      rw [← hk] at hw ⊢
      rw [← hw]
      refine Mem.read_congr fun x hx => hfr x ?_
      rintro ⟨t, ⟨i, he, rfl⟩, hxt⟩
      have hne : (qend d init b - nlog d init b).toNat ≠ gW i := by
        intro h
        have h' := congrArg (fun n : Nat => (n : Int)) h
        simp only [hk, hgW] at h'
        rw [hnd_def, hN] at h'
        push_cast at h'
        have := (i.1 : Nat).cast_nonneg (α := Int)
        omega
      exact Set.disjoint_left.mp (d.input_ids.span_disjoint (by exact_mod_cast hne)) hx hxt
    · -- the drafts
      intro _ j hj
      have hjB : j < d.BLOCK - 0 := by omega
      let i : Idx [d.BLOCK - 0] := (⟨j, hjB⟩, ())
      have hei : en i = true := decide_eq_true hj
      refine ⟨d32.get i, ?_, ?_⟩
      · have := hm' i hei
        rwa [hgW i] at this
      · rw [convRel_i64_i32_bmod (hd32 i)]
        have hr := hdv' i hei
        rw [read_keep (fun x hx => hR x (sub_R_draft d hx)) (hinR i hei), hgR i] at hr
        have hr' : init.read .i64
            (d.draft_tokens.ptr (ridx d init b * d.dstride + (j : Int))).addr = some (dv.get i) := hr
        rw [rdt64, rd64_of hr']
  · have hc : ¬ ((0 : I32).lt nd = true) := by
      rw [IntV.lt_eq, toInt_zero32, hnd]; simpa using hpos
    rw [ite_eq_right hc]
    exact safe_ret ⟨hQL, hQA, fun _ j hj => absurd hj (by omega)⟩

end Claims.CombineDraftTokens
