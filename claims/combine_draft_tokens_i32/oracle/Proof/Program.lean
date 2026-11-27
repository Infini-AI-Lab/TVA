import Proof.Tail

/-!
# `combine_draft_tokens_i32`: one program (source lines 75-118)

Program `b` loads its request-state row, `cu_num_logits[b]`, `cu_num_logits[b + 1]` and its query
end, stores its logits indices (lines 88-92), then runs `tailProg` (`Proof.Tail`).
`safe_program` proves it safe from `init` with the footprints of `Proof.Basic`.
-/

namespace Claims.CombineDraftTokens

open TVA

/-- The body of program `b`, as the model unfolds. -/
def bodyProg (d : Inst) (b : Nat) : Prog Unit := do
  let ri ← tl.load1 (d.idx_mapping.ptr 0 + (IntV.ofNat b : I32))
  let cs ← tl.load1 (d.cu_num_logits.ptr 0 + (IntV.ofNat b : I32))
  let ce ← tl.load1 (d.cu_num_logits.ptr 0 + (IntV.ofNat b : I32) + (1 : I32))
  let qe ← tl.load1 (d.query_start_loc.ptr 0 + (IntV.ofNat b : I32) + (1 : I32))
  let lv ← tl.convert .i32 .i64 ((qe - (ce - cs)) + tl.arange 0 d.BLOCK)
  tl.store (d.logits_indices.ptr 0 + cs + tl.arange 0 d.BLOCK) lv
    (some ((tl.arange 0 d.BLOCK).ltS (ce - cs)))
  tailProg d b ri cs ce qe

/-- The model's body of program `i` is `bodyProg`. -/
theorem body_eq (d : Inst) (i : Fin (launch d).size) :
    (model d).body ((launch d).ctx i) = bodyProg d i := by
  simp only [model, combine_sampled_and_draft_tokens_kernel, tl.program_id, launch_pid]
  rfl

/-- `b.ptr 0 + k + 1` is the pointer to element `k + 1`. -/
theorem ptr_add_one {dt : DType} (buf : Buf dt) {k : Nat} (hk : k < 2 ^ 31) :
    buf.ptr 0 + (IntV.ofNat k : I32) + (1 : I32) = buf.ptr ((k + 1 : Nat) : Int) := by
  rw [ptr_add_add buf _ _ (IntV.I32.toInt_ofNat hk) toInt_one32]
  push_cast; rfl

/-- The elements of `input_ids` program `b` writes lie in its query rows. -/
theorem ids_range {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.B)
    {t : Nat} (ht : t ∈ lastE d init b ∪ draftE d init b) :
    qsl d init b ≤ t ∧ (t : Int) < qend d init b := by
  obtain ⟨-, -, -, -, hNl, -, hq, -⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
  have hN1 := hd.hN
  rcases ht with ⟨hlast, ht⟩ | ⟨-, j, hj, ht⟩
  · have := hlast.2.1
    have hN : (1 : Int) ≤ d.N := by exact_mod_cast this
    constructor <;> omega
  · simp only [ndraft] at hj ht
    have := (j : Nat).cast_nonneg (α := Int)
    constructor <;> omega

/-- The elements of `logits_indices` program `b` writes lie in `[cu[b], cu[b + 1])`. -/
theorem log_range {d : Inst} {init : Mem} {b : Nat} {t : Nat} (ht : t ∈ logE d init b) :
    cuAt d init b ≤ t ∧ (t : Int) < cuAt d init (b + 1) := by
  obtain ⟨j, hj, ht⟩ := ht
  simp only [nlog] at hj
  have := (j : Nat).cast_nonneg (α := Int)
  constructor <;> omega

/-- The `input_ids` part of program `b`'s footprint lies in `input_ids`. -/
theorem ids_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.B) :
    d.input_ids.elems (lastE d init b) ∪ d.input_ids.elems (draftE d init b) ⊆ d.input_ids.bytes := by
  rw [← Buf.elems_union]
  refine d.input_ids.elems_sub_bytes fun t ht => ?_
  have := (ids_range hd hpre hb ht).2
  have := (qsl_bounds hpre hb).2
  omega

/-- The `logits_indices` part of program `b`'s footprint lies in `logits_indices`. -/
theorem log_sub {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.B) :
    d.logits_indices.elems (logE d init b) ⊆ d.logits_indices.bytes := by
  refine d.logits_indices.elems_sub_bytes fun t ht => ?_
  have := (log_range ht).2
  have := (cu_bounds hpre hb).2
  omega

/-- **One program** of the grid is safe from `init`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq]
  have hb : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
  generalize (i : Nat) = b at hb ⊢
  have hb31 : b < 2 ^ 31 := lt_trans hb hd.hgrid
  obtain ⟨-, -, -, -, hNl, hlB, hq, -⟩ := hpre.2.2.2.2.2.2.2.2.2.1 b hb
  obtain ⟨hq0, hq1⟩ := qsl_bounds hpre hb
  obtain ⟨hc0, hc1⟩ := cu_bounds hpre hb
  have hB20 := block_le hd
  unfold bodyProg
  rw [ptr_add_one d.cu_num_logits hb31, ptr_add_one d.query_start_loc hb31,
    ptr_add d.idx_mapping _ (IntV.I32.toInt_ofNat hb31),
    ptr_add d.cu_num_logits _ (IntV.I32.toInt_ofNat hb31)]
  have hidx : b < d.idx_mapping.len := lt_of_lt_of_le hb hd.hidx
  have hcu0 : b < d.cu_num_logits.len := by have := hd.hcu; omega
  have hcu1 : b + 1 < d.cu_num_logits.len := by have := hd.hcu; omega
  have hqs1 : b + 1 < d.query_start_loc.len := by have := hd.hqsl; omega
  -- lines 76-86
  refine safe_load1_buf d.idx_mapping hidx hpre.2.1
    ((d.idx_mapping.span_sub hidx).trans (sub_R_idx d)) fun ri hri => ?_
  refine safe_load1_buf d.cu_num_logits hcu0 hpre.2.2.2.2.2.2.2.1
    ((d.cu_num_logits.span_sub hcu0).trans (sub_R_cu d)) fun cs hcs => ?_
  refine safe_load1_buf d.cu_num_logits hcu1 hpre.2.2.2.2.2.2.2.1
    ((d.cu_num_logits.span_sub hcu1).trans (sub_R_cu d)) fun ce hce => ?_
  refine safe_load1_buf d.query_start_loc hqs1 hpre.2.2.2.1
    ((d.query_start_loc.span_sub hqs1).trans (sub_R_qsl d)) fun qe hqe => ?_
  have hri' : ri.toInt = ridx d init b := (rd64_of hri).symm
  have hcs' : cs.toInt = cuAt d init b := (rd32_of hcs).symm
  have hce' : ce.toInt = cuAt d init (b + 1) := (rd32_of hce).symm
  have hqe' : qe.toInt = qend d init b := (rd32_of hqe).symm
  have hqb : qend d init b < 2 ^ 31 := (rd32_bounds d.query_start_loc init ((b + 1 : Nat) : Int)).2
  have hnl : (ce - cs).toInt = nlog d init b :=
    IntV.toInt_sub hce' hcs' (sf32 (by simp only [nlog] at hNl; omega)
      (by simp only [nlog] at hlB ⊢; omega))
  have hls : (qe - (ce - cs)).toInt = qend d init b - nlog d init b :=
    IntV.toInt_sub hqe' hnl (sf32 (by omega) (by omega))
  -- lines 88-92
  refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun lv hlv => ?_
  let en : Idx [d.BLOCK - 0] → Bool := fun i => decide (((i.1 : Nat) : Int) < nlog d init b)
  have hen : ∀ i, en i = true → ((i.1 : Nat) : Int) < nlog d init b := fun i h =>
    of_decide_eq_true h
  have hmask : ∀ i, ((tl.arange 0 d.BLOCK).ltS (ce - cs)).get i = en i := by
    intro i
    rw [Tensor.get_ltS, IntV.lt_eq, arange_toInt (by omega), hnl]
  let g : Idx [d.BLOCK - 0] → Nat := fun i => (cuAt d init b + (i.1 : Nat)).toNat
  have hg : ∀ i, (g i : Int) = cuAt d init b + (i.1 : Nat) := fun i =>
    Int.toNat_of_nonneg (by positivity)
  have hin : ∀ i, en i = true → g i < d.logits_indices.len := by
    intro i he; have h1 := hen i he; have h2 := hg i; simp only [nlog] at h1; omega
  have hA : d.logits_indices.Addrs
      (tl.addrs (d.logits_indices.ptr 0 + cs + tl.arange 0 d.BLOCK) (some
        ((tl.arange 0 d.BLOCK).ltS (ce - cs)))) g en := by
    refine Buf.addrs_of_get _ hmask fun i _ => ?_
    rw [Tensor.get_ptrAdd, ptr_add d.logits_indices _ hcs', Buf.ptr_offset,
      arange_toInt (by omega), hg i]
  have hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j := by
    intro i j _ _ h
    have h' := congrArg (fun n : Nat => (n : Int)) h
    simp only [hg] at h'
    exact Prod.ext (Fin.ext (by exact_mod_cast (by linarith : ((i.1 : Nat) : Int) = (j.1 : Nat))))
      rfl
  have himg : Buf.img g en ⊆ logE d init b := by
    rintro t ⟨i, he, rfl⟩
    exact ⟨(i.1 : Nat), hen i he, hg i⟩
  refine safe_store_buf d.logits_indices hA hin hinj hpre.2.2.2.2.2.2.2.2.1
    ((d.logits_indices.elems_mono himg).trans Set.subset_union_left) fun m1 hm1 hfr1 hpres1 => ?_
  -- what the store leaves
  have hTl : d.logits_indices.elems (Buf.img g en) ⊆ d.logits_indices.bytes :=
    d.logits_indices.elems_sub_bytes (by rintro j ⟨i, he, rfl⟩; exact hin i he)
  have hR1 : ∀ x ∈ R d, m1 x = init x := by
    intro x hx
    refine hfr1 x fun hxT => ?_
    obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := hd.hlog
    have hxl := hTl hxT
    simp only [R, Set.mem_union] at hx
    rcases hx with (((((hx | hx) | hx) | hx) | hx) | hx) | hx
    · exact Set.disjoint_left.mp h1 hxl hx
    · exact Set.disjoint_left.mp h2 hxl hx
    · exact Set.disjoint_left.mp h3 hxl hx
    · exact Set.disjoint_left.mp h4 hxl hx
    · exact Set.disjoint_left.mp h5 hxl hx
    · exact Set.disjoint_left.mp h6 hxl hx
    · exact Set.disjoint_left.mp h7 hxl hx
  have hV1 : d.input_ids.Valid m1 :=
    Buf.valid_of_agree hpre.1 fun x hx => hpres1 x (hpre.1.2.2.2 x hx)
  have hQL1 : QL d init b m1 := by
    intro j hj
    have hjB : j < d.BLOCK - 0 := by omega
    let i : Idx [d.BLOCK - 0] := (⟨j, hjB⟩, ())
    have hei : en i = true := decide_eq_true hj
    refine ⟨lv.get i, ?_, ?_⟩
    · have := hm1 i hei
      rwa [hg i] at this
    · rw [ConvRel.toInt_i32_i64 (hlv i), Tensor.get_sadd]
      exact IntV.toInt_add hls (arange_toInt (by omega) i) (sf32 (by omega) (by omega))
  refine (safe_tail hd hpre E hb ri cs ce qe hri' hcs' hce' hqe' hR1 hV1 hQL1).mono
    (fun _ _ h => h) fun x hx => ⟨Set.subset_union_right hx, fun hxl => ?_⟩
  exact Set.disjoint_left.mp hd.hids.1 (ids_sub hd hpre hb hx) (hTl hxl)

end Claims.CombineDraftTokens
